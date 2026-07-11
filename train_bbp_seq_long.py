# !/usr/bin/env python3
import sys
import os
import numpy as np
import torch
from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.path.append('/home/ubuntu/workspace')
import argparse
from model.generator_long import LayoutGenerator
from model.bbp_dataset_long import lab_dataset
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, SGD

from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from collections import OrderedDict
# from warmup_scheduler import GradualWarmupScheduler
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as init
from model.bppEnv import data_fiter

order_max_len = 200
order_min_len = 5
sheet_max_len = 4
max_target_len = 450
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
seed = 12
torch.manual_seed(seed)  # torch的CPU随机性,为CPU设置随机种子
torch.cuda.manual_seed(seed)  # torch的GPU随机性,为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)
# scaler = GradScaler(enabled=False)
scaler = torch.amp.GradScaler('cuda', enabled=False)
parser = argparse.ArgumentParser(description='PyTorch on BPP with RL')
parser.add_argument('--epoch', default=120, type=int, help='Number of epochs')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--batch', type=int, default=40, help='Learning rate')

parser.add_argument('--data', type=str, default='./data/merge_data', help=' table of item in BPP')
parser.add_argument('--train', type=str, default='./data/real.txt', help=' table of item in BPP')
parser.add_argument('--test', type=str, default='./data/test.txt', help=' table of item in BPP')

parser.add_argument('--load_path', type=str, default='./checkpoints',
                    help='Path to load model parameters and optimizer state from')
parser.add_argument('--resume', type=bool, default=True, help='Resume from previous checkpoint file')
parser.add_argument('--resume_epoch', type=int, default=1500, help='Resume from previous checkpoint file')
# GPU
# parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
# parser.add_argument('--gpu', type=int, default=0, help='gpu')
# parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
# 如果是多机多卡的机器，WORLD_SIZE代表使用的机器数，RANK对应第几台机器
# 如果是单机多卡的机器，WORLD_SIZE代表有几块GPU，RANK和LOCAL_RANK代表第几块GPU

parser.add_argument('--local_rank', type=int, default=0, help='DDP parameter, do not modify')
parser.add_argument('--world_size', type=int, default=2, help='maximum number of dataloader workers')
parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')

# Network
parser.add_argument('--embedding_dim', type=int, default=1024, help='Embedding size')
parser.add_argument('--hiddens', type=int, default=1024, help='Number of hidden units')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout value')

args = parser.parse_args()
# args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
# scheduler_warmup = GradualWarmupScheduler(optim_generator, multiplier=1, total_epoch=5, after_scheduler=G_scheduler)
mse_loss = torch.nn.MSELoss()
cls_loss = torch.nn.CrossEntropyLoss(reduction='none')
cls_loss2 = torch.nn.CrossEntropyLoss()
bce_loss = torch.nn.BCEWithLogitsLoss()

train_set = lab_dataset(args.data, args.train, True, order_max_len, sheet_max_len, order_min_len, max_target_len)


# train_set = data_fiter(train_set)

def set_nan_inf_0(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                param.grad[torch.isnan(param.grad)] = 0.0
                param.data[torch.isnan(param.data)] = 0.0
            if torch.isinf(param.grad).any():
                param.grad[torch.isinf(param.grad)] = 0.0
                param.data[torch.isinf(param.data)] = 0.0


def ddp_setup(rank, world_size):
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:56789', rank=rank,
                            world_size=world_size)  # distributed backend
    torch.cuda.set_device(rank)


class Trainer(object):
    def __init__(self,
                 model: torch.nn.Module,
                 train_data: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 gpu_id: int
                 ) -> None:
        self.gpu_id = gpu_id
        self.train_loader = train_data
        # self.model = DDP(model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=True)
        self.model = DDP(model, device_ids=[gpu_id], output_device=gpu_id)
        # self.model = model
        self.optim = optimizer
        self.scheduler = scheduler
        # print('num batches: %d' % len(train_data)/4)

    def run_batch(self, data):
        self.model.train()
        input_seq, sheet, target_seq, n_typ_item, n_typ_sheet, target_len, edge, final_cut_ratio, item_records, local_cost = data
        batch_size = input_seq.shape[0]

        it_mask = torch.zeros(batch_size, order_max_len + 1)  # add begin node
        bn_mask = torch.zeros(batch_size, sheet_max_len)
        tg_mask = torch.zeros(batch_size, max_target_len + 1)  # add begin

        it_mask = it_mask.to(self.gpu_id, non_blocking=True).long()
        bn_mask = bn_mask.to(self.gpu_id, non_blocking=True).long()
        tg_mask = tg_mask.to(self.gpu_id, non_blocking=True).long()
        local_cost = local_cost.to(self.gpu_id, non_blocking=True)

        for b in range(batch_size):
            bn_mask[b, :n_typ_sheet[b]] = sheet[b, :n_typ_sheet[b], 0]
            it_mask[b, 1:n_typ_item[b] + 1] = input_seq[b, :n_typ_item[b], -1]
            tg_mask[b, :target_len[b] + 1] = 1  # including begin but not including end
        it_mask[:, 0] = 1

        input_seq = input_seq.to(self.gpu_id, non_blocking=True)
        sheet = sheet.to(self.gpu_id, non_blocking=True)
        target_seq = target_seq.to(self.gpu_id, non_blocking=True)
        edge = edge.to(self.gpu_id, non_blocking=True)

        # _target_seq = target_mask * target_seq
        '''
        for name, param in self.model.named_parameters():
            print(name)
            print(param[0,0:10])
            exit(1)
        '''
        with autocast(enabled=False):
            bin_id, cut_level_1, cut_level_2, cut_typ, lc, rc, lr, rr, level_1_lab, level_2_lab, pred_cost_num, pred_rate = self.model(
                input_seq,
                target_seq,
                sheet,
                it_mask,
                bn_mask,
                tg_mask,
                edge,
                item_records)

        # target_node_typ = target_node_typ.to(self.gpu_id, non_blocking=True)
        N = target_len  # 开始节点不用预测,结束节点要预测
        loss_cost = seq_loss3(pred_cost_num, local_cost, N)
        loss_rate = seq_loss2(pred_rate, final_cut_ratio, N)
        loss_bin = seq_loss(bin_id, target_seq[:, :, 0].long(), N)
        loss_cut_x = seq_loss(cut_level_1, level_1_lab, N)
        loss_cut_y = seq_loss(cut_level_2, level_2_lab, N)
        loss_cuts = loss_cut_x + loss_cut_y
        loss_cut_typ = seq_loss(cut_typ, target_seq[:, :, 3].long(), N)
        loss_lc = seq_loss(lc, target_seq[:, :, 4].long(), N)
        loss_lr = seq_loss(lr, target_seq[:, :, 5].long(), N)
        loss_rc = seq_loss(rc, target_seq[:, :, 6].long(), N)
        loss_rr = seq_loss(rr, target_seq[:, :, 7].long(), N)
        loss_next = loss_rr + loss_rc + loss_lr + loss_lc + loss_cut_typ + loss_cuts + loss_bin + loss_cost + loss_rate
        return loss_next, loss_bin.item(), loss_cuts.item(), loss_cut_typ.item(), loss_lc.item(), loss_lr.item(), loss_rc.item(), loss_rr.item(), loss_cost.item(), loss_rate.item()

    def train(self, end_epoch, start_epoch=0):
        for e in range(start_epoch, end_epoch):
            G_loss = []
            b_sz = len(next(iter(self.train_loader))[0])
            print(f"Epoch: {e} | GPU_{self.gpu_id} | batch_size: {b_sz}")
            # iterator = tqdm(data_loader, unit='Batch')
            self.train_loader.sampler.set_epoch(e)
            for i, input_data in enumerate(self.train_loader):
                torch.cuda.empty_cache()
                loss_next, loss_bin, loss_cuts, loss_cut_typ, loss_lc, loss_lr, loss_rc, loss_rr,loss_cost,loss_rate = self.run_batch(
                    input_data)
                g_loss = loss_next
                G_loss.append(g_loss.item())

                if i % 50 == 0:
                    print('GPU: %d' % self.gpu_id, i, 'total_loss: %.5f' % g_loss.item(),
                          'pos_loss: %.5f' % loss_cuts, 'cut_typ_loss: %.5f' % loss_cut_typ,
                          'lc_loss: %.5f' % loss_lc, 'bin_loss: %.5f' % loss_bin, 'rc_loss: %.5f' % loss_rc,'loss_cost: %.5f' % loss_cost,'loss_rate: %.5f' % loss_rate,
                          flush=True)
                self.optim.zero_grad()
                with torch.autograd.set_detect_anomaly(True):
                    # print(g_loss)
                    # set_nan_inf_0(self.model)
                    scaler.scale(g_loss).backward()
                    # g_loss.backward()
                    # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
                    scaler.step(self.optim)
                    scaler.update()
                    # self.optim.step()
                    '''
                    for name, param in model.named_parameters():
                        if param.grad is not None and name.find('cls') != -1:
                            print(name)
                            print(param.grad[param.grad == -torch.inf])
                    # exit(1)
                    '''
            print('epoch %d | gen_loss: %.5f, lr: %.5f' % (e, np.mean(G_loss), self.scheduler.get_last_lr()[0]),
                  flush=True)
            self.scheduler.step()
            # scheduler_warmup.step(e)
            # print(e, optim_generator.param_groups[0]['lr'])
            if e % 10 == 0 and self.gpu_id == 0:
                ck2 = {
                    'generator': get_inner_model(self.model).state_dict(),
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state_all()
                }
                torch.save(ck2, os.path.join(args.load_path, 'epoch-{}.pt'.format(e)))


def seq_loss(preds, targets, target_lens):
    batch = preds.shape[0]
    loss = 0
    for b in range(batch):
        pred = preds[b, :target_lens[b]]
        loss += cls_loss2(pred, targets[b, :target_lens[b]].squeeze())
    loss /= batch
    return loss


def seq_loss2(preds, targets, target_lens):
    batch = preds.shape[0]
    loss = 0
    for b in range(batch):
        pred = preds[b, :target_lens[b] + 1]
        tgt = torch.zeros(pred.shape)
        tgt[...] = targets[b].item()
        tgt = tgt.to(pred.device)
        loss += mse_loss(pred, tgt)
    loss /= batch
    return loss


def dec2bin(num, n_bits):
    tensors = []
    for i in range(num.shape[0]):
        binary_str = bin(num[i])[2:]  # 去除前缀'0b'
        binary_str = binary_str.zfill(n_bits)
        binary_tensor = torch.tensor([int(bit) for bit in binary_str])
        tensors.append(binary_tensor.unsqueeze(0))
    tensors = torch.cat(tensors, dim=0)
    return tensors.float()


def seq_loss3(preds, targets, target_lens):
    batch = preds.shape[0]
    loss = 0
    for b in range(batch):
        pred = preds[b, :target_lens[b]]
        obs = targets[b, :target_lens[b]]
        tgt = obs + pred
        real = obs[-1]
        target = torch.zeros(target_lens[b], 1).to(pred.device)
        label = torch.zeros(target_lens[b], 1).to(pred.device)
        label[...] = torch.log(real)
        target[:, 0] = torch.log(tgt[:, 0])
        loss += mse_loss(target, label)
    loss /= batch
    return loss


def get_inner_model(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def load_model(args):
    generator = LayoutGenerator(embedding_dim=args.embedding_dim, hidden_dim=args.hiddens, order_max_len=order_max_len,
                                sheet_max_len=sheet_max_len, dropout=args.dropout)

    if args.resume:
        pth = os.path.join(args.load_path, 'epoch-{}.pt'.format(args.resume_epoch))
        if os.path.exists(pth):
            ck = torch.load(pth, map_location=lambda storage, loc: storage, weights_only=True)
            # ck = torch.load(pth, map_location='cuda:{}'.format(rank))
        else:
            print('resumed model missing!')
            exit(1)
        torch.set_rng_state(ck['rng_state'])
        '''
        if args.use_gpu:
            torch.cuda.set_rng_state_all(ck['cuda_rng_state'])
        '''
        new_state_dict = OrderedDict()
        pretrained = ck['generator']
        for k, v in generator.state_dict().items():
            if k in pretrained.keys():
                new_state_dict[k] = pretrained[k]
            else:
                new_state_dict[k] = v
        generator.load_state_dict(new_state_dict, strict=True)
        print('finish loading pretrained model!!')

    optim = Adam(generator.parameters(), lr=args.lr)
    # optim_generator = SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=500, gamma=0.1)
    return generator, optim, scheduler


def main(rank: int, world_size: int, batch_size: int):
    ddp_setup(rank, world_size)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        pin_memory=True,
        # num_workers=2,
        drop_last=True,
        sampler=DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    )

    model, optimizer, scheduler = load_model(args)
    trainer = Trainer(model.to(rank), train_loader, optimizer, scheduler, rank)
    trainer.train(args.epoch)
    dist.destroy_process_group()


if __name__ == "__main__":
    print('num training cases: %d' % (len(train_set)))
    print('num batches: %d' % (len(train_set) / (args.batch * args.world_size)))
    mp.spawn(main, args=(args.world_size, args.batch), nprocs=args.world_size, join=True)
