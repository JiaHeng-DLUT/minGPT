import numpy as np


data_path = '../../Fruit_Fly_Groups/Notebooks/data/submission_data.npy'
submission_data = np.load(data_path, allow_pickle=True).item()
seqs = submission_data['sequences']
id_list = list(seqs.keys())
n = len(id_list)
print(n)
id_list.sort()
f = open(f'meta_info/meta_info_test.txt', 'w')
for id in id_list:
    f.write(f'{id}\n')
f.close()
