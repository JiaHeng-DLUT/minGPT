"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import os

from tqdm import tqdm

import torch
from torch.utils.data.dataloader import DataLoader

from data.fly_dataset import FlyNormDataset
from eval import Evaluator
from model import GPT, GPT1Config
from utils import set_random_seed


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
    }
    val_dataset = {
        'data_path': '../../Fruit_Fly_Groups/Notebooks/data/user_train.npy',
        'meta_path': 'meta_info/meta_info_val_0.txt',
        'num_frame': clip_frames,
        'total_frame': total_frames,
    }
    batch_size = 32
    num_workers = 4

    # optimization parameters
    max_epochs = 100
    learning_rate = 1e-5
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)

    # checkpoint settings
    ckpt_dir = f'./experiments/fly/43_lr_regression_bs32_lr_1e-5'
    # CUDA_VISIBLE_DEVICES=0 python 43_lr_regression_bs32_lr_1e-5.py > experiments/fly/log/43_lr_regression_bs32_lr_1e-5.log

    evaluator_config = {
            'num_seeds': 3,
            'num_subtasks': 2,
            'num_process': 3,
            'lr_list': [1e-6, 1e-5, 1e-4, \
                        1e-3, 1e-2, 1e-1, \
                        1e0, 1e1, 1e2],
            'batch_size': 8192,
            'input_dim': 256,
            'output_dim': 2,
    }

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        
        config.evaluator_config['num_samples'] = len(test_dataset) * config.clip_frames
        self.evaluator = Evaluator(config.evaluator_config)

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
            for it, data in pbar:

                # place data on the correct device
                x = data['keypoints'].to(self.device)           #(b, clip_frame, 528)
                y = data['labels'].to(self.device).long()       #(b, 3, clip_frame)
                pos = data['pos']

                # forward the model
                with torch.set_grad_enabled(is_train):
                    feat, losses = model(x, pos, y)
                    loss = 0
                    for k, v in losses.items():
                        loss += v

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    print(f'epoch {epoch+1}, iter {it}, lr {lr:e}, train loss: {loss.item():.5f}', end='')
                    for k, v in losses.items():
                        print(f', {k}: {v.item():.5f}', end='')
                    print()
                else:
                    feat = feat.view(-1, feat.shape[-1]).cpu()
                    feats.append(feat)
                    label = y.transpose(-1, -2)
                    label = label.reshape(-1, label.shape[-1]).cpu()
                    labels.append(label)

            if not is_train:
                feats = torch.cat(feats, dim=0)
                labels = torch.cat(labels, dim=0)
                return (feats, labels)

        self.tokens = 0 # counter used for learning rate decay
        # if self.test_dataset is not None:
        #     best_metric = self.evaluator.eval(*run_epoch('test'))
        best_metric = 0
        for epoch in range(config.max_epochs):
            run_epoch('train')
            torch.cuda.empty_cache()
            if self.test_dataset is not None:
                metric = self.evaluator.eval(*run_epoch('test'))
                torch.cuda.empty_cache()
                if metric > best_metric:
                    best_metric = metric
                    self.save_checkpoint(epoch + 1)


if __name__ == '__main__':
    set_random_seed(0)
    config = TrainerConfig()
    train_set = FlyNormDataset(config.train_dataset)
    val_set = FlyNormDataset(config.val_dataset)
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
