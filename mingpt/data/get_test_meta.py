import numpy as np
from tqdm import tqdm


data_path = '../../Fruit_Fly_Groups/Notebooks/data/frame_number_map.npy'
frame_number_map = np.load(data_path, allow_pickle=True).item()
id_list = list(frame_number_map.keys())
n = len(id_list)
print(n)
f = open(f'meta_info/meta_info_test.txt', 'w')
i = 0
for id in tqdm(id_list):
    f.write(f'{id}\n')
    print(frame_number_map[id])
    (st, ed) = frame_number_map[id]
    if st != i:
        assert(0)
    i += 4500
f.close()
