import numpy as np
import torch
import torch.nn.functional as F


# submission = np.load('experiments/fly/01_max_epoch_100/epoch1_submission.npy', allow_pickle=True).item()
# map_submission = submission['frame_number_map']
# map_gt = np.load('../../Fruit_Fly_Groups/Notebooks/data/frame_number_map.npy', allow_pickle=True).item()
# for id in map_gt:
#     print(map_gt[id])
#     print(map_submission[id])
#     assert(map_gt[id] == map_submission[id])


submission1 = np.load('experiments/fly/01_max_epoch_100/epoch1_submission.npy', allow_pickle=True).item()
submission2 = np.load('experiments/fly/07_lr_1e-3/epoch2_submission.npy', allow_pickle=True).item()
embeddings1 = torch.from_numpy(submission1['embeddings'])
embeddings2 = torch.from_numpy(submission2['embeddings'])
print(embeddings1.shape, embeddings2.shape)
embeddings1 = F.normalize(embeddings1, dim=1)
embeddings2 = F.normalize(embeddings2, dim=1)
ans = (embeddings1 * embeddings2).sum(1)
print(ans)
