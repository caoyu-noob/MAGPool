import numpy as np
import json
import os

# dir_path = 'PROTEINS'
# files = os.listdir(dir_path)
# prefix = 'multihead_simple'
# part_files = []
# for f in files:
#     if f[:len(prefix)] == prefix:
#         part_files.append(f)
# best_acc, best_std = 0, 0
# best_file = None
# for f in part_files:
#     with open(os.path.join(dir_path, f), 'r') as f:
#         d = json.load(f)
#     cur_acc = np.mean(d['val_acc'])
#     if cur_acc > best_acc:
#         best_file = f
#         best_acc = cur_acc
#         best_std = np.std(d['val_acc'])
# print(best_file)
# print(best_acc)
# print(best_std)

with open('COLLAB/SAG-global_standard_global_complex_max_headnum2_hid128_lr0.0001_pr0.5_dr0.5_results.json', 'r') as f:
    d = json.load(f)
# with open('COLLAB/SortPool_standard_global_complex_max_headnum2_hid128_lr0.0005_pr0.5_dr0.5_results.json1', 'r') as f:
#     d1 = json.load(f)
# d['val_acc'].extend(d1['val_acc'])
# d['val_loss'].extend(d1['val_loss'])
print(np.mean(d['val_acc']))
print(np.std(d['val_acc']))