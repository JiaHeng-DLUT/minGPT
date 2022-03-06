import numpy as np


submission = np.load('experiments/fly/01_max_epoch_100/epoch1_submission.npy', allow_pickle=True).item()
map_submission = submission['frame_number_map']
map_gt = np.load('../../Fruit_Fly_Groups/Notebooks/data/frame_number_map.npy', allow_pickle=True).item()
for id in map_gt:
    print(map_gt[id])
    print(map_submission[id])
    assert(map_gt[id] == map_submission[id])
