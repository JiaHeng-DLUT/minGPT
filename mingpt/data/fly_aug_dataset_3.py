import numpy as np
import time
import torch
import torch.utils.data as data


class fly_aug_dataset_3(data.Dataset):
    """Fruit fly dataset.
    """

    def __init__(self, opt):
        super(fly_aug_dataset_3, self).__init__()
        self.opt = opt
        data_path = opt['data_path']
        meta_path = opt['meta_path']

        data = np.load(data_path, allow_pickle=True).item()
        self.seqs = data['sequences']

        id_list = open(meta_path).readlines()
        self.id_list = [id.strip() for id in id_list]
        
        self.num_frame = opt['num_frame']                               # Number of frames per clip
        self.num_clip = int(opt['total_frame'] / opt['num_frame'])      # Number of clips per video
        
        self.mean = torch.Tensor([
            -0.8837589, 1.7955586, -0.8858594, 1.7961606, -0.8892626, 1.8271109, \
            -0.88458323, 1.8214903, -0.89488745, 1.8221262, -0.89360124, 1.8224231, \
            -0.886063, 1.8210831, -0.8897023, 1.8135841, -0.8757437, 1.8006829, \
            -0.8854544, 1.8108734, -0.87983066, 1.8117129, -0.88819385, 1.8164563, \
            -0.888663, 1.8111625, -0.86693615, 1.8128147, -0.8521162, 1.8044478, \
            -0.858751, 1.7963856, -0.8687342, 1.7942374, -0.87776256, 1.7925133, \
            -0.8758628, 1.8111782, -0.8851568, 1.8104229, -0.0018544069, 0.0062475177, \
            2.8482492, 0.97616994, 1.9980444, 5.7338033, 0.36938184, 4.468912])
        self.std = torch.Tensor([
            13.121616, 12.856946, 13.112699, 12.845017, 13.587678, 13.342501, 13.513889, \
            13.256663, 13.501413, 13.263471, 13.478333, 13.239276, 13.488188, 13.232512, \
            13.344328, 13.089211, 13.207274, 12.939953, 13.398516, 13.141108, 13.39832, \
            13.12874, 13.388674, 13.139851, 13.378667, 13.129087, 13.654897, 13.401184, \
            13.444521, 13.173474, 13.220815, 12.958844, 13.18577, 12.934439, 13.40502, \
            13.153501, 13.6389, 13.408308, 13.372433, 13.122187, 0.70575887, 0.7082579, \
            0.27629352, 0.091889516, 0.33484602, 2.4920604, 0.06724139, 4.7354803
        ])

        self.horizon_flip_prob = opt['horizon_flip_prob'] if 'horizon_flip_prob' in opt else None
        print(f'horizon_flip_prob: {self.horizon_flip_prob}')
        self.vertical_flip_prob = opt['vertical_flip_prob'] if 'vertical_flip_prob' in opt else None
        print(f'vertical_flip_prob: {self.vertical_flip_prob}')
        self.h_translation_prob = opt['h_translation_prob'] if 'h_translation_prob' in opt else None
        print(f'h_translation_prob: {self.h_translation_prob}')
        self.v_translation_prob = opt['v_translation_prob'] if 'v_translation_prob' in opt else None
        print(f'v_translation_prob: {self.v_translation_prob}')
        self.max_translation = opt['max_translation'] if 'max_translation' in opt else None
        print(f'max_translation: {self.max_translation}')
        self.rotation_prob = opt['rotation_prob'] if 'rotation_prob' in opt else None
        print(f'rotation_prob: {self.rotation_prob}')
        print('---')

    def __getitem__(self, index):
        ret = {}

        id = self.id_list[index // self.num_clip]
        ret.update({'id': id})
        pos = index % self.num_clip * self.num_frame
        ret.update({'pos': pos})

        keypoints = torch.from_numpy(self.seqs[id]['keypoints'][pos : pos + self.num_frame])    # (num_frame, 11, 24, 2)
        # data augmentations
        if self.horizon_flip_prob is not None:
            if np.random.uniform() < self.horizon_flip_prob:
                # (-x, y), (-cos, sin)
                keypoints[:, :, :21, 0] = -keypoints[:, :, :21, 0]
        if self.vertical_flip_prob is not None:
            if np.random.uniform() < self.vertical_flip_prob:
                # (x, -y), (cos, -sin)
                keypoints[:, :, :21, 1] = -keypoints[:, :, :21, 1]
        if self.h_translation_prob is not None:
            if np.random.uniform() < self.h_translation_prob:
                h_translation = np.random.uniform(low=-self.max_translation, high=self.max_translation)
                keypoints[:, :, :20, 0] += h_translation
        if self.v_translation_prob is not None:
            if np.random.uniform() < self.v_translation_prob:
                v_translation = np.random.uniform(low=-self.max_translation, high=self.max_translation)
                keypoints[:, :, :20, 1] += v_translation
        if self.rotation_prob is not None:
            # if np.random.uniform() < self.rotation_prob:
            if np.random.uniform() < 1:
                rotation = np.random.uniform(low=-np.pi, high=np.pi)
                R = torch.Tensor([
                    [np.cos(rotation), -np.sin(rotation)],
                    [np.sin(rotation), np.cos(rotation)]
                ])
                keypoints[:, :, :20] = keypoints[:, :, :20] @ R
                angle = torch.acos(keypoints[:, :, 20, 0])
                angle -= rotation
                keypoints[:, :, 20, 0] = torch.cos(angle)
                keypoints[:, :, 20, 1] = torch.sin(angle)

        keypoints = torch.flatten(keypoints, 2)         # (num_frame, 11, 48)
        mask = ~(torch.isnan(keypoints).long().sum(-1).bool())
        ret.update({'mask': mask})
        keypoints = (keypoints - self.mean) / self.std
        # keypoints = torch.flatten(keypoints, 1)         # (num_frame, 528)
        keypoints = torch.nan_to_num(keypoints, nan=0)
        ret.update({'keypoints': keypoints})

        if 'annotations' in self.seqs[id]:
            labels = torch.from_numpy(self.seqs[id]['annotations'][:, pos : pos + self.num_frame])
            labels = torch.nan_to_num(labels, nan=-100)
            ret.update({'labels': labels})

        return ret

    def __len__(self):
        return len(self.id_list) * self.num_clip


if __name__ == '__main__':
    # train
    train_dataset = {
        'data_path': '../../Fruit_Fly_Groups/Notebooks/data/user_train.npy',
        'meta_path': 'meta_info/meta_info_train_0.txt',
        'num_frame': 50,
        'total_frame': 4500,
    }
    dataset = fly_aug_dataset_3(train_dataset)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=1)
    data_st = time.time()
    for i, data in enumerate(dataloader):
        print(i)
        break

    # # val
    # val_dataset = {
    #     'data_path': '../../Fruit_Fly_Groups/Notebooks/data/user_train.npy',
    #     'meta_path': 'meta_info/meta_info_val_0.txt',
    #     'num_frame': 150,
    #     'total_frame': 4500,
    # }
    # dataset = fly_aug_dataset_3(val_dataset)
    # dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    # labels = []
    # for i, data in enumerate(dataloader):
    #     label = data['labels']
    #     labels.append(label)
    # labels = torch.cat(labels, dim=0)
    # print(1, labels.shape)
    # for i in range(3):
    #     label = labels[:, i]
    #     print(i, label.shape)
    #     print(0, torch.count_nonzero(label == 0))
    #     print(1, torch.count_nonzero(label == 1))
