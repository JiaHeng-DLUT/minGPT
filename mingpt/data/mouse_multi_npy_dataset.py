import numpy as np
import torch
import torch.utils.data as data

from mingpt.utils import get_root_logger


class MouseMultiNpyDataset(data.Dataset):
    """Mouse dataset.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            data_path (str): Path for data file (npy).
            meta_path (str): Path for meta information file.

            num_frames (int): Window size for input frames.
            total_frames (int): Total frames in a video.
            use_label (bool): Whether to get label ot not.
    """

    def __init__(self, opt):
        super(MouseMultiNpyDataset, self).__init__()
        self.opt = opt
        data_path_list = opt['data_path']
        meta_path = opt['meta_path']
        # Number of frames per clip
        self.num_frames = opt['num_frames']
        # Number of clips per video
        self.num_clips = int(opt['total_frames'] / opt['num_frames'])
        self.use_label = opt['use_label']

        data = {}
        for data_path in data_path_list:
            data = {**data, **np.load(data_path, allow_pickle=True).item()['sequences']}
        self.seqs = data

        id_list = open(meta_path).readlines()
        self.id_list = [id.strip() for id in id_list]

        logger = get_root_logger()
        self.horizon_flip_prob = opt.get('horizon_flip_prob', None)
        logger.info(f'horizon_flip_prob: {self.horizon_flip_prob}')
        self.vertical_flip_prob = opt.get('vertical_flip_prob', None)
        logger.info(f'vertical_flip_prob: {self.vertical_flip_prob}')
        self.h_translation_prob = opt.get('h_translation_prob', None)
        logger.info(f'h_translation_prob: {self.h_translation_prob}')
        self.v_translation_prob = opt.get('v_translation_prob', None)
        logger.info(f'v_translation_prob: {self.v_translation_prob}')
        self.max_translation = opt.get('max_translation', None)
        logger.info(f'max_translation: {self.max_translation}')
        self.rotation_prob = opt.get('rotation_prob', None)
        logger.info(f'rotation_prob: {self.rotation_prob}')

        self.mean = torch.Tensor([
            190.3901306712963, 357.6238096064815, 190.81475405092593,
            357.2520934027778, 191.1934693287037, 357.65426678240743,
            190.56950046296296, 357.4511219907407, 189.413884375,
            357.47640960648147, 189.6061107638889, 357.9850104166667,
            190.56748472222222, 357.1351216435185, 190.87997465277778,
            356.34231909722223, 191.1901173611111, 357.5504020833333,
            192.6579152777778, 356.81606215277776, 198.9482880787037,
            355.8314724537037, 203.36558900462964, 356.6498457175926])
        self.std = torch.Tensor([
            194.05687477080573, 241.31872051279225, 191.59390506539074,
            242.14703018833234, 191.69054375525658, 242.07700071351056,
            191.33704747903374, 241.74542698218335, 192.2191264213517,
            241.53357008536608, 192.40013406703352, 241.38879449163636,
            189.41473275114058, 241.2476368866387, 188.698006643059,
            240.28791408211558, 188.91058571092623, 240.0673255642363,
            186.7104325901908, 238.99194411212062, 180.7910322980137,
            231.89481621653016, 175.28417599534558, 226.19533972655105])

    def __getitem__(self, index):
        ret = {}

        id = self.id_list[index // self.num_clips]
        ret.update({'id': id})

        pos = index % self.num_clips * self.num_frames
        ret.update({'pos': pos})

        # (num_frames, num_animals, num_keypoints, 2)
        keypoints = torch.from_numpy(
            self.seqs[id]['keypoints'][pos: pos + self.num_frames])

        # data augmentations
        if self.horizon_flip_prob is not None:
            if np.random.uniform() < self.horizon_flip_prob:
                # (-x, y), (-cos, sin)
                keypoints[:, :, :, 0] = -keypoints[:, :, :, 0]
        if self.vertical_flip_prob is not None:
            if np.random.uniform() < self.vertical_flip_prob:
                # (x, -y), (cos, -sin)
                keypoints[:, :, :, 1] = -keypoints[:, :, :, 1]
        if self.h_translation_prob is not None:
            if np.random.uniform() < self.h_translation_prob:
                h_translation = np.random.uniform(
                    low=-self.max_translation, high=self.max_translation)
                keypoints[:, :, :, 0] += h_translation
        if self.v_translation_prob is not None:
            if np.random.uniform() < self.v_translation_prob:
                v_translation = np.random.uniform(
                    low=-self.max_translation, high=self.max_translation)
                keypoints[:, :, :, 1] += v_translation
        if self.rotation_prob is not None:
            # if np.random.uniform() < self.rotation_prob:
            if np.random.uniform() < 1:
                rotation = np.random.uniform(low=-np.pi, high=np.pi)
                R = torch.Tensor([
                    [np.cos(rotation), -np.sin(rotation)],
                    [np.sin(rotation), np.cos(rotation)]
                ])
                keypoints = keypoints @ R

        # (num_frames, num_animals, num_keypoints * 2)
        keypoints = torch.flatten(keypoints, 2)
        # (num_frames, num_animals)
        mask = ~(torch.isnan(keypoints).long().sum(-1).bool())
        ret.update({'mask': mask})

        keypoints = (keypoints - self.mean) / self.std
        keypoints = torch.nan_to_num(keypoints, nan=0)
        ret.update({'keypoints': keypoints})

        if self.use_label:
            labels = torch.from_numpy(
                self.seqs[id]['annotations'][:, pos: pos + self.num_frames]).T
            labels = torch.nan_to_num(labels, nan=-100)
            ret.update({'labels': labels})

        return ret

    def __len__(self):
        return len(self.id_list) * self.num_clips


if __name__ == '__main__':
    # # train
    # train_dataset = {
    #     'data_path': '../Mouse_Triplets/Notebooks/data/user_train.npy',
    #     'meta_path': 'mingpt/data/meta_info/mouse/mouse_meta_info_train_0.txt',
    #     'num_frames': 50,
    #     'total_frames': 1800,
    #     'use_label': False,
    # }
    # dataset = MouseDataset(train_dataset)
    # dataloader = data.DataLoader(
    #     dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=1)
    # for i, data in enumerate(dataloader):
    #     id = data['id']
    #     keypoints = data['keypoints']
    #     mask = data['mask']
    #     pos = data['pos']
    #     print(len(id), keypoints.shape, mask.shape, pos.shape)
    #     # 32 torch.Size([32, 50, 3, 24]) torch.Size([32, 50, 3]) torch.Size([32])
    #     break

    # val
    val_dataset = {
        'data_path': '../Mouse_Triplets/Notebooks/data/user_train.npy',
        'meta_path': 'mingpt/data/meta_info/mouse/mouse_meta_info_val_0.txt',
        'num_frames': 50,
        'total_frames': 1800,
        'use_label': True,
    }
    dataset = MouseDataset(val_dataset)
    dataloader = data.DataLoader(
        dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=1)
    for i, data in enumerate(dataloader):
        id = data['id']
        keypoints = data['keypoints']
        mask = data['mask']
        pos = data['pos']
        labels = data['labels']
        print(len(id), keypoints.shape, mask.shape, pos.shape, labels.shape)
        # 32 torch.Size([32, 50, 3, 24]) torch.Size([32, 50, 3]) torch.Size([32]) torch.Size([32, 50, 2])
        break
