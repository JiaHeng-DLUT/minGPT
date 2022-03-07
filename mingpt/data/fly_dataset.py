import numpy as np
import torch
import torch.utils.data as data


class FlyDataset(data.Dataset):
    """Fruit fly dataset.
    """

    def __init__(self, opt):
        super(FlyDataset, self).__init__()
        self.opt = opt
        data_path = opt['data_path']
        meta_path = opt['meta_path']

        data = np.load(data_path, allow_pickle=True).item()
        self.seqs = data['sequences']
        id_list = open(meta_path).readlines()
        self.id_list = [id.strip() for id in id_list]
        self.num_frame = opt['num_frame']                               # Number of frames per clip
        self.num_repeat = int(opt['total_frame'] / opt['num_frame'])    # Number of clips per video 

    def __getitem__(self, index):
        id = self.id_list[index // self.num_repeat]
        keypoints = torch.from_numpy(self.seqs[id]['keypoints'])    # (4500, 11, 24, 2)
        keypoints = torch.flatten(keypoints, 1)                     # (4500, 528)
        keypoints = torch.nan_to_num(keypoints, nan=0)
        if 'annotations' in self.seqs[id]:
            labels = torch.from_numpy(self.seqs[id]['annotations'])
            labels = torch.nan_to_num(labels, nan=-100)
        else:
            labels = None
        pos = index % self.num_repeat * self.num_frame
        
        if labels is None:
            return {
                'keypoints': keypoints[pos : pos + self.num_frame],
                'pos': pos,
                'id': id,
            }
        else:
            return {
                'keypoints': keypoints[pos : pos + self.num_frame],
                'labels': labels[:, pos : pos + self.num_frame],
                'pos': pos,
                'id': id,
            }

    def __len__(self):
        return len(self.id_list) * self.num_repeat



class FlyNormDataset(data.Dataset):
    """Fruit fly dataset.
    """

    def __init__(self, opt):
        super(FlyNormDataset, self).__init__()
        self.opt = opt
        data_path = opt['data_path']
        meta_path = opt['meta_path']

        data = np.load(data_path, allow_pickle=True).item()
        self.seqs = data['sequences']
        id_list = open(meta_path).readlines()
        self.id_list = [id.strip() for id in id_list]
        self.num_frame = opt['num_frame']                               # Number of frames per clip
        self.num_repeat = int(opt['total_frame'] / opt['num_frame'])    # Number of clips per video
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

    def __getitem__(self, index):
        id = self.id_list[index // self.num_repeat]
        keypoints = torch.from_numpy(self.seqs[id]['keypoints'])    # (4500, 11, 24, 2)
        keypoints = torch.flatten(keypoints, 2)                     # (4500, 11, 48)
        keypoints = (keypoints - self.mean) / self.std
        keypoints = torch.flatten(keypoints, 1)                     # (4500, 528)
        keypoints = torch.nan_to_num(keypoints, nan=0)
        if 'annotations' in self.seqs[id]:
            labels = torch.from_numpy(self.seqs[id]['annotations'])
            labels = torch.nan_to_num(labels, nan=-100)
        else:
            labels = None
        pos = index % self.num_repeat * self.num_frame
        
        if labels is None:
            return {
                'keypoints': keypoints[pos : pos + self.num_frame],
                'pos': pos,
                'id': id,
            }
        else:
            return {
                'keypoints': keypoints[pos : pos + self.num_frame],
                'labels': labels[:, pos : pos + self.num_frame],
                'pos': pos,
                'id': id,
            }

    def __len__(self):
        return len(self.id_list) * self.num_repeat


if __name__ == '__main__':
    # train
    train_dataset = {
        'data_path': '../../Fruit_Fly_Groups/Notebooks/data/user_train.npy',
        'meta_path': 'meta_info/meta_info_train_0.txt',
        'num_frame': 150,
        'total_frame': 4500,
    }
    dataset = FlyNormDataset(train_dataset)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    for i, data in enumerate(dataloader):
        print(i)

    # val
    # dataset = FlyDataset(opt['datasets']['val'])
    # dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    # for i, data in enumerate(dataloader):
    #     print(i)
