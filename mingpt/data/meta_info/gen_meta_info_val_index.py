import random
import numpy as np
random.seed(0)


index_list = []
for i in range(6):
    index = list(range(43 * 4500))
    random.shuffle(index)
    index_list.append(index)
np.savetxt('fly_meta_info_val_0_index.txt', np.array(index_list), fmt='%s')
