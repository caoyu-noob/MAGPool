import os

dataset = 'DD'

with open(os.path.join('data', dataset, 'raw', dataset + '_graph_indicator.txt'), 'r') as f:
    data = f.readlines()
node_nums = []
count = 1
cur_indicator = data[0]
for i in range(1, len(data)):
    if data[i] != cur_indicator:
        node_nums.append(count)
        count = 1
        cur_indicator = data[i]
    else:
        count += 1
node_nums.sort()
print(node_nums[int(len(node_nums) * 0.1)])