import math
import time
import shutil
import os 

import numpy as np
import torch
import torchvision

from .dataloader import build_dataloader
from .model import build_model
from .utils import AverageMeter
from .utils import concat_all_gather


class Pretrainer:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

        # build dataloader
        self.train_loader, self.memory_loader, self.test_loader = build_dataloader(self.cfg)
        self.total_steps = self.cfg.epochs * len(self.train_loader)
        self.warmup_steps = self.cfg.warmup_epochs * len(self.train_loader)
        
        # build model
        self.model = build_model(self.cfg)
        self.logger.info(f'{self.model}')

        # build optimizer
        self.optimizer = self.build_optimizer()
        self.scaler = torch.cuda.amp.GradScaler()

        # build loss
        self.loss = getattr(self, self.cfg.loss)

    def build_optimizer(self):
        self.init_lr = self.cfg.base_lr * self.cfg.whole_batch_size / 256

        optim_params = self.model.module.parameters()
        optimizer = torch.optim.SGD(optim_params, self.init_lr,
                                    momentum=self.cfg.momentum,
                                    weight_decay=self.cfg.weight_decay)
        
        return optimizer
   
    def adjust_lr(self, optimizer, step):
        max_lr = self.init_lr
        min_lr = 1e-3 * self.init_lr
        if step < self.warmup_steps:
            lr = (max_lr - min_lr) * step / self.warmup_steps + min_lr
        else:
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos((step - self.warmup_steps) * np.pi / self.total_steps))
        for param_group in optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = self.init_lr
            else:
                param_group['lr'] = lr
        
        return lr

    def adjust_mm(self, base_mm, step, schedule='cos'):
        if schedule == 'cos':
            return 1 - (1 - base_mm) * (np.cos(np.pi * step / self.total_steps) + 1) / 2
        elif schedule == 'const':
            return base_mm

    def resume(self, resume_path):
        ckpt = torch.load(resume_path, map_location='cpu')
        self.model.load_state_dict(ckpt['model'], strict=False)
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scaler.load_state_dict(ckpt['scaler'])

        self.start_epoch = ckpt['epoch'] + 1
        self.step = ckpt['step']
        self.F = ckpt['F']

        if self.F is not None:
            self.F = self.F.cuda()

    def pretrain(self):
        self.last_saved_ckpt = None
        self.start_epoch, self.step = 0, 0
        self.F = None # EMA for correlation matrix

        # resume if required
        if self.cfg.resume_path is not None:
            self.resume(self.cfg.resume_path)

            # knn eval if required
            # be careful to do evaluation during training, the memory cost is too large
            knn_eval = getattr(self.cfg, 'knn_eval', False)
            if knn_eval:
                self.logger.info(f'{self.cfg.resume_path}')
                self.logger.info(f'knn: {self.knn_eval(self.model)}')
                return 0

        # begin training
        self.logger.info(f'Begin training, start_epoch:{self.start_epoch}, step:{self.step}')
        for epoch in range(self.start_epoch, self.cfg.epochs):
            if torch.distributed.is_available():
                self.train_loader.sampler.set_epoch(epoch)

            # collect epoch statistics
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4f')
            pos_sims = AverageMeter('Pos Sim', ':.4f')
           
            # switch to train mode
            self.model.train()
            
            end = time.time()
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            for i, data in enumerate(self.train_loader):
                # adjust lr and mm
                lr = self.adjust_lr(self.optimizer, self.step)
                mm = self.adjust_mm(self.cfg.base_momentum, self.step)
                self.step += 1

                with torch.cuda.amp.autocast():
                    images = data[0]
                    x1 = images[0].cuda(non_blocking=True)
                    x2 = images[1].cuda(non_blocking=True)
                    data_time.update(time.time() - end)

                    # forward
                    z1, z2, z1m, z2m = self.model(x1, x2, mm=mm)

                    # normalize
                    z1 = torch.nn.functional.normalize(z1)
                    z2 = torch.nn.functional.normalize(z2)
                    z1m = torch.nn.functional.normalize(z1m)
                    z2m = torch.nn.functional.normalize(z2m)

                    # compute loss
                    loss, pos_sim = self.loss(z1, z2, z1m, z2m)

                    # exit if loss nan
                    if torch.any(torch.isnan(loss)):
                        print(f'{torch.cuda.current_device()} {loss}') 
                    return_flag = torch.tensor([0]).cuda()
                    if torch.isnan(loss):
                        return_flag = torch.tensor([1]).cuda()
                    torch.distributed.all_reduce(return_flag)
                    if return_flag:
                        self.logger.info(f"exit with loss value: {loss}")
                        return -1

                losses.update(loss.item(), images[0].size(0))
                pos_sims.update(pos_sim.item(), images[0].size(0))
                
                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
                if i==0 or (i+1) % self.cfg.print_freq == 0:
                    self.logger.info(f'Epoch: [{epoch}/{self.cfg.epochs}]  ' \
                                     f'iter: {i+1}/{len(self.train_loader)}  ' \
                                     f'{str(batch_time)}  ' \
                                     f'{str(data_time)}  ' \
                                     f'{str(losses)} ' \
                                     f'{str(pos_sims)} ' \
                                     f'lr: {lr} ' \
                                     f'mm: {mm}')

            # save model
            if torch.distributed.get_rank() == 0:
                if (epoch+1) % self.cfg.save_freq == 0:
                    ckpt_name = os.path.join(self.cfg.work_dir, 'checkpoint_{}.pth.tar'.format(epoch))
                else:
                    ckpt_name = os.path.join(self.cfg.work_dir, 'latest.pth.tar')
                self.logger.info('saving model')
                torch.save({'model': self.model.state_dict(),
                           'optimizer': self.optimizer.state_dict(),
                           'scaler': self.scaler.state_dict(),
                           'step': self.step,
                           'epoch': epoch,
                           'F': self.F,}, ckpt_name)
    
    def knn_eval(self, model):
        net = model.module.encoder
        projector = model.module.projector
        net.eval()
        projector.eval()
        classes = len(self.memory_loader.dataset.classes)
        total_top1, total_top5, total_num, feature_bank, target_bank = 0.0, 0.0, 0, [], []
        with torch.no_grad():
            # generate feature bank
            i = 0
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            for data, target in self.memory_loader:
                feature = net(data.cuda(non_blocking=True))
                feature = torch.nn.functional.normalize(feature, dim=1)
                feature_bank.append(feature.clone())
                target_bank.append(target.cuda().clone())
                i += 1
                
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            target_bank = torch.cat(target_bank, dim=0).contiguous()
            
            tensors_gather = [torch.ones_like(feature_bank)
                for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensors_gather, feature_bank, async_op=False)

            feature_bank = torch.cat(tensors_gather, dim=-1)
            
            tensors_gather = [torch.ones_like(target_bank)
                for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensors_gather, target_bank, async_op=False)

            target_bank = torch.cat(tensors_gather, dim=0)

            # loop test data to predict the label by weighted knn search
            i = 0
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            for data, target in self.test_loader:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                feature = net(data)
                feature = torch.nn.functional.normalize(feature, dim=1)

                pred_labels = self.knn_predict(feature, feature_bank, target_bank, classes, self.cfg.knn_k, self.cfg.knn_t)

                total_num += data.size(0)
                total_top1 += (pred_labels[:, 0] == target).float().sum().item()
                i += 1
            
            total_num = torch.tensor(total_num).cuda()
            total_top1 = torch.tensor(total_top1).cuda()
            torch.distributed.all_reduce(total_num)
            torch.distributed.all_reduce(total_top1)

        return total_top1.item() / total_num.item() * 100
    
    def knn_predict(self, feature, feature_bank, feature_labels, classes, knn_k, knn_t):
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = torch.mm(feature, feature_bank)
        # [B, K]
        sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
        # [B, K]
        sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / knn_t).exp()

        # counts for each class
        one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        return pred_labels

    def loss_unigrad(self, z1, z2, z1m, z2m):
        # calculate correlation matrix
        tmp_F = (torch.mm(z1m.t(), z1m) + torch.mm(z2m.t(), z2m)) / (2*z1m.shape[0])
        torch.distributed.all_reduce(tmp_F)
        tmp_F = tmp_F / torch.distributed.get_world_size()
        if self.F is None:
            self.F = tmp_F.detach()
        else:
            self.F = self.cfg.rho * self.F + (1 - self.cfg.rho) * tmp_F.detach()

        # compute grad & loss
        grad1 = -z2m + self.cfg.lambd*torch.mm(z1, self.F)
        loss1 = (grad1.detach() * z1).sum(-1).mean()
        
        grad2 = -z1m + self.cfg.lambd*torch.mm(z2, self.F)
        loss2 = (grad2.detach() * z2).sum(-1).mean()

        loss = 0.5 * (loss1 + loss2)
        
        # compute positive similarity, just for observation
        pos_sim1 = torch.einsum('nc,nc->n', [z1, z2m]).mean().detach()
        pos_sim2 = torch.einsum('nc,nc->n', [z2, z1m]).mean().detach()
        pos_sim = 0.5 * (pos_sim1 + pos_sim2)
        
        return loss, pos_sim

    def loss_simclr_grad_mm(self, z1, z2, z1m, z2m):
        # gather neg samples
        neg_samples = torch.cat([z1m, z2m], dim=0)
        neg_gather = [torch.ones_like(neg_samples) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(neg_gather, neg_samples, async_op=False)
        all_neg_samples = torch.cat(neg_gather, dim=0)

        # loss1 from z1 and z2m
        pos_term = -z2m

        weight = z1@all_neg_samples.t()
        weight = torch.nn.functional.softmax(weight/self.cfg.moco_t, dim=-1)
        neg_term = weight@all_neg_samples

        grad1 = (pos_term + neg_term) / self.cfg.moco_t
        loss1 = (z1 * grad1.detach()).sum(-1).mean()

        # loss2 from z2 and z1m
        pos_term = -z1m

        weight = z2@all_neg_samples.t()
        weight = torch.nn.functional.softmax(weight/self.cfg.moco_t, dim=-1)
        neg_term = weight@all_neg_samples

        grad2 = (pos_term + neg_term) / self.cfg.moco_t
        loss2 = (z2 * grad2.detach()).sum(-1).mean()

        loss = 0.5 * (loss1 + loss2)
        
        # compute positive similarity, just for observation
        pos_sim1 = torch.einsum('nc,nc->n', [z1, z2m]).mean().detach()
        pos_sim2 = torch.einsum('nc,nc->n', [z2, z1m]).mean().detach()
        pos_sim = 0.5 * (pos_sim1 + pos_sim2)

        return loss, pos_sim

    def loss_byol_directpred_grad(self, z1, z2, z1m, z2m):
        # calculate F & wp
        tmp_F = (torch.mm(z1m.t(), z1m) + torch.mm(z2m.t(), z2m)) / (2*z1m.shape[0])
        torch.distributed.all_reduce(tmp_F)
        tmp_F = tmp_F / torch.distributed.get_world_size()
        if self.F is None:
            self.F = tmp_F.detach()
        else:
            self.F = self.cfg.rho * self.F + (1 - self.cfg.rho) * tmp_F.detach()
        u, s, v = torch.svd(self.F.float(), compute_uv=True)
        max_s = torch.max(s)
        sqrt_F = torch.mm(u, torch.mm(torch.diag(torch.sqrt(s)), v.t())).half()
        s = s / torch.max(s)
        s = torch.sqrt(s) + self.cfg.eps
        s = s.clamp(1e-4)
        wp = torch.mm(u, torch.mm(torch.diag(s), v.t())).half()
        eps_mat = self.cfg.eps**2*torch.eye(z1.shape[1]).cuda()

        # loss1 is made up of z1 & z2m
        # calculate common denominator
        common_denominator = torch.linalg.norm(torch.mm(z1, wp), dim=-1).unsqueeze(-1) # (n, 1)

        # calculate positive term
        pos_term = - torch.mm(z2m, wp) # (n, c)

        # calculate lambda
        neg_denominator = torch.einsum('nc,nc->n', [torch.mm(z1, self.F/max_s + eps_mat), z1]).unsqueeze(-1) # (n, 1)
        neg_numerator = torch.einsum('nc,nc->n', [torch.mm(z1, wp), z2m]).unsqueeze(-1) # (n, 1)
        neg_lambda = neg_numerator / neg_denominator

        # calculate negative term
        neg_term = torch.mm(z1, self.F/max_s + eps_mat) # (n, c)

        grad1 = 1/common_denominator * (pos_term + neg_lambda * neg_term) # (n, c)
        loss1 = (z1 * grad1.detach()).sum(-1).mean()

        # loss2 is made up of z2 & z1m
        # calculate common denominator
        common_denominator = torch.linalg.norm(torch.mm(z2, wp), dim=-1).unsqueeze(-1) # (n, 1)

        # calculate positive term
        pos_term = - torch.mm(z1m, wp) # (n, c)

        # calculate lambda
        neg_denominator = torch.einsum('nc,nc->n', [torch.mm(z2, self.F/max_s + eps_mat), z2]).unsqueeze(-1) # (n, 1)
        neg_numerator = torch.einsum('nc,nc->n', [torch.mm(z2, wp), z1m]).unsqueeze(-1) # (n, 1)
        neg_lambda = neg_numerator / neg_denominator

        # calculate first negative term
        neg_term = torch.mm(z2, self.F/max_s + eps_mat) # (n, c)

        grad2 = 1/common_denominator * (pos_term + neg_lambda * neg_term) # (n, c)
        loss2 = (z2 * grad2.detach()).sum(-1).mean()
        loss = 0.5 * (loss1 + loss2)

        # compute positive similarity, just for observation
        pos_sim1 = torch.einsum('nc,nc->n', [z1, z2m]).mean().detach()
        pos_sim2 = torch.einsum('nc,nc->n', [z2, z1m]).mean().detach()
        pos_sim = 0.5 * (pos_sim1 + pos_sim2)
        
        return loss, pos_sim

    def loss_barlow_twins_grad_mm(self, z1, z2, z1m, z2m):
        # gather all samples
        z1_bank = concat_all_gather(z1)
        z2_bank = concat_all_gather(z2)
        z1m_bank = concat_all_gather(z1m)
        z2m_bank = concat_all_gather(z2m)

        N = self.cfg.whole_batch_size
        # loss1 from z1 and z2m
        pos = -z2m
        neg = (z2m @ z2m_bank.T) @ z1_bank / N

        grad1 = 2 * (pos + self.cfg.lambd * neg)
        loss1 = (z1 * grad1.detach()).sum(-1).mean()

        # loss2 from z2 and z1m
        pos = -z1m
        neg = (z1m @ z1m_bank.T) @ z2_bank / N

        grad2 = 2 * (pos + self.cfg.lambd * neg)
        loss2 = (z2 * grad2.detach()).sum(-1).mean()
        loss = 0.5 * (loss1 + loss2)

        # compute positive similarity, just for observation
        pos_sim1 = torch.einsum('nc,nc->n', [z1, z2m]).mean()
        pos_sim2 = torch.einsum('nc,nc->n', [z2, z1m]).mean()
        pos_sim = 0.5*(pos_sim1 + pos_sim2)

        return loss, pos_sim