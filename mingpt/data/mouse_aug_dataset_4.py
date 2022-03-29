import numpy as np
import time
import torch
import torch.utils.data as data


class mouse_aug_dataset_4(data.Dataset):
    """Fruit fly dataset.
    """

    def __init__(self, opt):
        super(mouse_aug_dataset_4, self).__init__()
        self.opt = opt
        meta_path = opt['meta_path']
        
        self.seqs = {}
        for data_path in opt['data_path']:
            data = np.load(data_path, allow_pickle=True).item()
            self.seqs = {**self.seqs, **data['sequences']}
        print(len(self.seqs))

        id_list = open(meta_path).readlines()
        self.id_list = [id.strip() for id in id_list]
        
        self.num_frame = opt['num_frame']                               # Number of frames per clip
        self.num_clip = int(opt['total_frame'] / opt['num_frame'])      # Number of clips per video
        
        self.mean = torch.Tensor([
            190.3901306712963, 357.6238096064815, 190.81475405092593, 357.2520934027778, 191.1934693287037, 357.65426678240743, \
            190.56950046296296, 357.4511219907407, 189.413884375, 357.47640960648147, 189.6061107638889, 357.9850104166667, \
            190.56748472222222, 357.1351216435185, 190.87997465277778, 356.34231909722223, 191.1901173611111, 357.5504020833333, \
            192.6579152777778, 356.81606215277776, 198.9482880787037, 355.8314724537037, 203.36558900462964, 356.6498457175926])
        self.std = torch.Tensor([
            194.05687477080573, 241.31872051279225, 191.59390506539074, 242.14703018833234, 191.69054375525658, 242.07700071351056, \
            191.33704747903374, 241.74542698218335, 192.2191264213517, 241.53357008536608, 192.40013406703352, 241.38879449163636, \
            189.41473275114058, 241.2476368866387, 188.698006643059, 240.28791408211558, 188.91058571092623, 240.0673255642363, \
            186.7104325901908, 238.99194411212062, 180.7910322980137, 231.89481621653016, 175.28417599534558, 226.19533972655105])

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
                keypoints[:, :, :, 0] = -keypoints[:, :, :, 0]
        if self.vertical_flip_prob is not None:
            if np.random.uniform() < self.vertical_flip_prob:
                # (x, -y), (cos, -sin)
                keypoints[:, :, :, 1] = -keypoints[:, :, :, 1]
        if self.h_translation_prob is not None:
            if np.random.uniform() < self.h_translation_prob:
                h_translation = np.random.uniform(low=-self.max_translation, high=self.max_translation)
                keypoints[:, :, :, 0] += h_translation
        if self.v_translation_prob is not None:
            if np.random.uniform() < self.v_translation_prob:
                v_translation = np.random.uniform(low=-self.max_translation, high=self.max_translation)
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

        keypoints = torch.flatten(keypoints, 2)         # (num_frame, 3, 24)
        mask = ~(torch.isnan(keypoints).long().sum(-1).bool()).view(-1)
        ret.update({'mask': mask})
        keypoints = (keypoints - self.mean) / self.std
        # keypoints = torch.flatten(keypoints, 1)         # (num_frame, 72)
        keypoints = torch.nan_to_num(keypoints, nan=0)
        ret.update({'keypoints': keypoints})

        if not self.opt['train']:
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
        'num_frame': 150,
        'total_frame': 4500,
    }
    dataset = mouse_aug_dataset_4(train_dataset)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=1)
    data_st = time.time()
    for i, data in enumerate(dataloader):
        x = data['keypoints'].to(0)           #(b, clip_frame, 528)
        y = data['labels'].to(0).long()       #(b, 3, clip_frame)
        pos = data['pos']
        data_ed = time.time()
        print(f'total: {data_ed - data_st}')
        data_st = time.time()

    # # val
    # val_dataset = {
    #     'data_path': '../../Fruit_Fly_Groups/Notebooks/data/user_train.npy',
    #     'meta_path': 'meta_info/meta_info_val_0.txt',
    #     'num_frame': 150,
    #     'total_frame': 4500,
    # }
    # dataset = mouse_aug_dataset_4(val_dataset)
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
