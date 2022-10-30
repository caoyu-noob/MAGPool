# MAGPool: Multi-Subspace Attention Graph Pooling
The implementation for paper **Multi-Subspace Attention Graph Pooling**

![MAGPool Framework](https://github.com/caoyu-noob/MAGPool/blob/main/framework.JPG)

## Requirements
1. Python==3.7
2. torch
3. torch_scatter
4. torch_geometric
5. matplotlib

## Datasets

We used torch_geometric to manage the datasets, it will automatically download the corresponding dataset 
once you run the code and put it under `\data`

## Run

The main entrance of this project is in `main.py`.

The default settings of `main.py` correspond to the MAGPool under hierarchical architecture we proposed in our paper.

So you can simply repeat the experiments via 

```
python main.py
```

It also contains baseline models and other variants of MAGPool we mentioned in our paper. 
If you want to try other baselines or variants, please change the value of `--model_type` into the corresponding ones.

The default dataset is `DD`, you can change it to other datasets by fill different values to this field.

If you need to change other configurations, please check `L216-L260` in `main.py` and make changes.
