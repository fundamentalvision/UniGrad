import copy
import math 
import torch
import torch.nn as nn
import torchvision.models as models


def build_model(cfg):
    model = SiameseNet(cfg)
    if torch.distributed.is_available():
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda()
        
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[torch.cuda.current_device()],
                                                          find_unused_parameters=True)
    else:
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    
    return model


class SiameseNet(nn.Module):
    """
    Build a Siamese model.
    Some codes are borrowed from MoCo & SimSiam.
    """

    def __init__(self, cfg):
        super(SiameseNet, self).__init__()
        self.cfg = cfg
        
        zero_init_residual = getattr(self.cfg, 'zero_init_residual', True)
        net = getattr(models.resnet, cfg.arch)(zero_init_residual=zero_init_residual)

        # build online branch
        self.encoder = nn.Sequential(*list(net.children())[:-1] + [nn.Flatten(1)])
        self.projector = _projection_mlp(self.cfg.projector_input_dim,
                                            self.cfg.projector_hidden_dim,
                                            self.cfg.projector_output_dim)

        # build target branch
        self.momentum_encoder = copy.deepcopy(self.encoder)
        self.momentum_projector = copy.deepcopy(self.projector)
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self, mm):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        for param_q, param_k in zip(self.projector.parameters(), self.momentum_projector.parameters()):
            param_k.data = param_k.data * mm + param_q.data * (1. - mm)

    def forward(self, x1, x2, mm):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https:/ /arxiv.org/abs/2011.10566 for detailed notations
        """  
        # online branch
        y1 = self.encoder(x1)
        y2 = self.encoder(x2)
        z1 = self.projector(y1)
        z2 = self.projector(y2)
        
        # target branch
        with torch.no_grad():
            self._momentum_update_key_encoder(mm)
            y1m = self.momentum_encoder(x1)
            y2m = self.momentum_encoder(x2)
            z1m = self.momentum_projector(y1m)
            z2m = self.momentum_projector(y2m)

        return z1, z2, z1m, z2m


def _projection_mlp(in_dims: int,
                    h_dims: int,
                    out_dims: int,
                    bias: bool = False) -> nn.Sequential:
    """Projection MLP. The original paper's implementation has 3 layers, with
    BN applied to its hidden fc layers but no ReLU on the output fc layer.
    The CIFAR-10 study used a MLP with only two layers.

    Args:
        in_dims:
            Input dimension of the first linear layer.
        h_dims:
            Hidden dimension of all the fully connected layers.
        out_dims:
            Output Dimension of the final linear layer.

    Returns:
        nn.Sequential:
            The projection head.
    """
    l1 = nn.Sequential(nn.Linear(in_dims, h_dims, bias=bias),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l2 = nn.Sequential(nn.Linear(h_dims, h_dims, bias=bias),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l3 = nn.Sequential(nn.Linear(h_dims, out_dims))

    projection = nn.Sequential(l1, l2, l3)
    
    return projection