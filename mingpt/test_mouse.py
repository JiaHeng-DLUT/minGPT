"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import os

from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from data.mouse_aug_dataset_2 import mouse_aug_dataset_2
from model2 import GPT, GPT1Config
from utils.misc import set_random_seed

class TesterConfig:
    # model
    input_dim = 12 * 2
    output_dim = 128
    total_frames = 1800
    clip_frames = 50
    num_animals = 3

    # data
    test_dataset = {
        'data_path': '../../Mouse_Triplets/Notebooks/data/submission_data.npy',
        'meta_path': 'meta_info/mouse_meta_info_test.txt',
        'num_frame': clip_frames,
        'total_frame': total_frames,
    }
    batch_size = int(total_frames // clip_frames)
    num_workers = 4

    # checkpoint setting
    ckpt_path = f'./experiments/fly/m11_n_embd/epoch1.pth'
    feat_path = ckpt_path.replace('.pth', '_submission_wo_mask.npy')
    # CUDA_VISIBLE_DEVICES=0 python test_mouse.py

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


def validate_submission(submission, submission_clips):
    if not isinstance(submission, dict):
      print("Submission should be dict")
      return False

    if 'frame_number_map' not in submission:
      print("Frame number map missing")
      return False

    if 'embeddings' not in submission:
        print('Embeddings array missing')
        return False
    elif not isinstance(submission['embeddings'], np.ndarray):
        print("Embeddings should be a numpy array")
        return False
    elif not len(submission['embeddings'].shape) == 2:
        print("Embeddings should be 2D array")
        return False
    elif not submission['embeddings'].shape[1] <= 256:
        print("Embeddings too large, max allowed is 256")
        return False
    elif not isinstance(submission['embeddings'][0, 0], np.float32):
        print(f"Embeddings are not float32")
        return False

    
    total_clip_length = 0
    for key in submission_clips['sequences']:
        start, end = submission['frame_number_map'][key]
        clip_length = submission_clips['sequences'][key]['keypoints'].shape[0]
        total_clip_length += clip_length
        if not end-start == clip_length:
            print(f"Frame number map for clip {key} doesn't match clip length")
            return False
            
    if not len(submission['embeddings']) == total_clip_length:
        print(f"Emebddings length doesn't match submission clips total length")
        return False

    
    if not np.isfinite(submission['embeddings']).all():
        print(f"Emebddings contains NaN or infinity")
        return False
    
    print("All checks passed")
    return True


class Tester:

    def __init__(self, model, test_dataset, config):
        state_dict = torch.load(config.ckpt_path)
        print(state_dict.keys())
        for k in state_dict:
            if k.endswith('mask'):
                print(state_dict[k])
        for k in state_dict:
            if k.endswith('mask'):
                state_dict[k] = torch.ones_like(state_dict[k])
        for k in state_dict:
            if k.endswith('mask'):
                print(state_dict[k])
        model.load_state_dict(state_dict)
        self.model = model
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    @torch.no_grad()
    def test(self):
        model, config = self.model, self.config

        is_train = False
        model.train(is_train)
        model.eval()
        data = self.test_dataset
        loader = DataLoader(data, shuffle=False, pin_memory=True,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)

        pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
        feats = []
        frame_number_map = {}
        for it, data in pbar:
            # place data on the correct device
            x = data['keypoints'].to(self.device)           #(b, clip_frame, 528)
            pos = data['pos']
            mask = data['mask'].to(self.device).long()
            id = data['id'][0]
            # print(pos, id)
            if id not in frame_number_map:
                st = it * self.config.total_frames
                ed = st + self.config.total_frames
                frame_number_map[id] = (st, ed)
            # forward the model
            with torch.set_grad_enabled(is_train):
                feat = model(x, pos, mask, y=None).view(-1, self.config.output_dim).cpu()
            feats.append(feat)
        feats = torch.cat(feats, dim=0).numpy()
        # print(1, feats.shape)
        # for k in frame_number_map:
        #     print(k, frame_number_map[k])
        submission_dict = {
            "frame_number_map": frame_number_map, 
            "embeddings": feats
        }
        submission_clips = np.load(self.config.test_dataset['data_path'], allow_pickle=True).item()
        validate_submission(submission_dict, submission_clips)
        np.save(self.config.feat_path, submission_dict)


if __name__ == '__main__':
    set_random_seed(0)
    config = TesterConfig()
    test_set = mouse_aug_dataset_2(config.test_dataset)
    print(len(test_set))

    gpt_config = GPT1Config(block_size=config.total_frames, 
        input_dim=config.input_dim, 
        output_dim=config.output_dim, 
        num_tokens=config.clip_frames,
        num_animals=config.num_animals)
    model = GPT(gpt_config)
    print(model)
    # x = torch.randn((2, 4500, 528))
    # print(model(x).shape)

    tester = Tester(model, test_set, config)
    tester.test()
