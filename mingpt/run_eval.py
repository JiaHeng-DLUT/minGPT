"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import os

from tqdm import tqdm

import torch
import torch.optim as optim
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
    ckpt_dir = f'./experiments/fly/45_lr_regression_lr_cosine'
    start_epoch = 11
    end_epoch = 46
    # CUDA_VISIBLE_DEVICES=0 python run_eval.py > experiments/fly/log/eval_45_lr_regression_lr_cosine_11_46.log

    evaluator_config = {
            'num_seeds': 3,
            'num_subtasks': 2,
            'num_process': 1,
            'lr_list': [1e-6, 1e-5, 1e-4, \
                        1e-3, 1e-2, 1e-1, \
                        1e0, 1e1, 1e2],
            'batch_size': 1,
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

        self.device = torch.cuda.current_device()

    def eval(self):
        config = self.config

        def run_epoch(epoch):
            print(f'epoch: {epoch}!!!')
            # Dataset
            loader = DataLoader(self.test_dataset, shuffle=False, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)
            pbar = enumerate(loader)
            # Model
            self.model.load_state_dict(torch.load(f'{config.ckpt_dir}/epoch{epoch}.pth'))
            model = torch.nn.DataParallel(self.model).to(self.device)
            model.eval()

            feats = []
            labels = []
            with torch.set_grad_enabled(False):
                for it, data in pbar:
                    # place data on the correct device
                    x = data['keypoints'].to(self.device)           #(b, clip_frame, 528)
                    y = data['labels'].to(self.device).long()       #(b, 3, clip_frame)
                    pos = data['pos']
                    # forward the model
                    feat, losses = model(x, pos, y)
                    feat = feat.view(-1, feat.shape[-1])
                    feats.append(feat)
                    label = y.transpose(-1, -2)
                    label = label.reshape(-1, label.shape[-1])
                    labels.append(label)
                model = model.to('cpu')
                torch.cuda.empty_cache()
                feats = torch.cat(feats, dim=0)
                labels = torch.cat(labels, dim=0)
                return (feats, labels)

        for epoch in range(config.start_epoch, config.end_epoch):
            self.evaluator.eval(*run_epoch(epoch))


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
    trainer.eval()
