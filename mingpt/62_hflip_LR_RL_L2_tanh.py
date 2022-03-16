"""
CosineAnnealingLR(optimizer, T_max=7000, eta_min=1e-7)
"""
import os
import time
import torch
import torch.optim as optim
import wandb

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from data.fly_aug_dataset_2 import fly_aug_dataset_2
from eval import Evaluator
from model import GPT, GPT1Config
from utils import set_random_seed


wandb.init(project="fly", entity="jiaheng")


class TrainerConfig:
    # model
    input_dim = 528
    output_dim = 256
    total_frames = 4500
    clip_frames = 150

    # data
    train_dataset = {
        'data_path': '../../Fruit_Fly_Groups/Notebooks/data/user_train.npy',
        'meta_path': 'meta_info/meta_info_train_0.txt',
        'num_frame': clip_frames,
        'total_frame': total_frames,
        'horizon_flip_prob': 0.5,
    }
    val_dataset = {
        'data_path': '../../Fruit_Fly_Groups/Notebooks/data/user_train.npy',
        'meta_path': 'meta_info/meta_info_val_0.txt',
        'num_frame': clip_frames,
        'total_frame': total_frames,
    }
    batch_size = 32
    num_workers = 4

    # optimizer
    max_epochs = 100
    learning_rate = 1e-5
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1      # only applied on matmul weights

    # evaluation
    evaluator_config = {
        'num_seeds': 3,
        'num_subtasks': 2,
        'num_process': 1,
        'lr_list': [1e-6, 1e-5, 1e-4,
                    1e-3, 1e-2, 1e-1,
                    1e0, 1e1, 1e2],
        'batch_size': 1,
        'input_dim': 256,
        'output_dim': 2,
    }

    # checkpoint settings
    ckpt_dir = f'./experiments/fly/62_hflip_LR_RL_L2_tanh'
    # CUDA_VISIBLE_DEVICES=0 python 62_hflip_LR_RL_L2_tanh.py > experiments/fly/log/62_hflip_LR_RL_L2_tanh.log

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        config.evaluator_config['num_samples'] = len(test_dataset) * config.clip_frames
        self.evaluator = Evaluator(config.evaluator_config)

        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self, epoch):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        print(f'Saving {self.config.ckpt_dir}/epoch{epoch}.pth')
        os.makedirs(self.config.ckpt_dir, exist_ok=True)
        torch.save(raw_model.state_dict(), f'{self.config.ckpt_dir}/epoch{epoch}.pth')

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, 'module') else model
        optimizer = raw_model.configure_optimizers(config)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=7000, eta_min=1e-7)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)

            feats = []
            labels = []
            data_st = time.time()
            for it, data in pbar:

                # place data on the correct device
                x = data['keypoints'].to(self.device)           #(b, clip_frame, 528)
                y = data['labels'].to(self.device).long()       #(b, 3, clip_frame)
                pos = data['pos']

                # forward the model
                data_ed = time.time()
                iter_st = data_ed
                with torch.set_grad_enabled(is_train):
                    feat, losses = model(x, pos, y)

                if is_train:
                    loss = 0
                    for k, v in losses.items():
                        loss += v
                        wandb.log({k: v.item()})
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    scheduler.step()
                    # report progress
                    iter_ed = time.time()
                    data_time = data_ed - data_st
                    iter_time = iter_ed - iter_st
                    print(f'epoch: {epoch+1}, iter: {it}, lr: {scheduler.get_lr()[0]:e}, time (data): {iter_time + data_time:.3f} ({data_time:.3f}), l_total: {loss.item():.5f}', end='')
                    for k, v in losses.items():
                        print(f', {k}: {v.item():.5f}', end='')
                    print()
                    data_st = time.time()
                else:
                    feat = feat.view(-1, feat.shape[-1])
                    feats.append(feat)
                    label = y.transpose(-1, -2)
                    label = label.reshape(-1, label.shape[-1])
                    labels.append(label)

            if not is_train:
                feats = torch.cat(feats, dim=0)
                labels = torch.cat(labels, dim=0)
                return (feats, labels)

        if self.test_dataset is not None:
            best_metric = self.evaluator.eval(*run_epoch('test'))
        self.save_checkpoint(0)
        for epoch in range(config.max_epochs):
            run_epoch('train')
            if self.test_dataset is not None:
                metric = self.evaluator.eval(*run_epoch('test'))
                wandb.log({'metric': metric})
                if metric > best_metric:
                    best_metric = metric
                    self.save_checkpoint(epoch + 1)


if __name__ == '__main__':
    set_random_seed(0)
    config = TrainerConfig()
    wandb.config = {
        "learning_rate": config.learning_rate,
        "epochs": config.max_epochs,
        "batch_size": config.batch_size,
    }
    train_set = fly_aug_dataset_2(config.train_dataset)
    val_set = fly_aug_dataset_2(config.val_dataset)
    print(len(train_set), len(val_set))
    # train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=1, pin_memory=True,)
    # val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=1, pin_memory=True,)
    # print(len(train_loader), len(val_loader))

    gpt_config = GPT1Config(block_size=config.total_frames,
                            input_dim=config.input_dim,
                            output_dim=config.output_dim,
                            num_tokens=config.clip_frames)
    model = GPT(gpt_config)
    print(model)
    # x = torch.randn((2, 4500, 528))
    # print(model(x).shape)

    trainer = Trainer(model, train_set, val_set, config)
    trainer.train()
