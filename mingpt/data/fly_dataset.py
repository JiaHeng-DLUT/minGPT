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


if __name__ == '__main__':
    # train
    train_dataset = {
        'data_path': '../../Fruit_Fly_Groups/Notebooks/data/user_train.npy',
        'meta_path': 'meta_info/meta_info_train_0.txt',
        'num_frame': 150,
        'total_frame': 4500,
    }
    dataset = FlyDataset(train_dataset)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    for i, data in enumerate(dataloader):
        print(i, data[-2], data[-1])

    # val
    # dataset = FlyDataset(opt['datasets']['val'])
    # dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    # for i, data in enumerate(dataloader):
    #     print(i)
