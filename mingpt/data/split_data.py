import numpy as np
import random


seed = 0
# seed = 666
# seed = 999
random.seed(seed)
data_path = '../../Mouse_Triplets/Notebooks/data/user_train.npy'
user_train = np.load(data_path, allow_pickle=True).item()
seqs = user_train['sequences']
id_list = list(seqs.keys())
n = len(id_list)
print(n)
id_list.sort()
# print(id_list[:10])
random.shuffle(id_list)
# print(id_list[:10])
num_train = int(n * 0.9)
print(num_train)
f = open(f'meta_info/mouse_meta_info_train_{seed}.txt', 'w')
for id in id_list[:num_train]:
    f.write(f'{id}\n')
f.close()
f = open(f'meta_info/mouse_meta_info_val_{seed}.txt', 'w')
for id in id_list[num_train:]:
    f.write(f'{id}\n')
f.close()
