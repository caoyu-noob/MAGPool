import torch
import math
from torch_scatter import scatter_add

def topk(x, ratio, batch):
    batch_size = 1
    if len(x.size()) != 0:
        batch_size = x.size(0)
    num_nodes = scatter_add(batch.new_ones(batch_size), batch, dim=0)
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1),
         num_nodes.cumsum(dim=0)[:-1]], dim=0)

    index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

    min_val = x.min().item() - 1
    dense_x = x.new_full((batch_size * max_num_nodes,), min_val)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    _, perm = dense_x.sort(dim=-1, descending=True)

    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)

    k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
    mask = [
        torch.arange(k[i], dtype=torch.long, device=x.device) +
        i * max_num_nodes for i in range(batch_size)
    ]
    mask = torch.cat(mask, dim=0)

    perm = perm[mask]

    return perm

def generate_sub_head_nums(feature_num, head_num):
    sub_nums = []
    while feature_num / head_num < 5 and head_num > 1:
        head_num -= 1
    if feature_num % head_num == 0:
        sub_num = int(math.floor(feature_num / head_num))
        sub_nums = [sub_num] * head_num
    else:
        floor_sub_num = int(math.floor(feature_num / head_num))
        rest_head_num = feature_num - floor_sub_num * head_num
        for i in range(head_num):
            if rest_head_num > 0:
                sub_nums.append(floor_sub_num + 1)
                rest_head_num -= 1
            else:
                sub_nums.append(floor_sub_num)
    return sub_nums