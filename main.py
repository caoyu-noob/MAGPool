import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric import utils
from networks import Net
from networks import MAGPoolGCN, MAGPoolGCNGlobal, MAGPoolGCNNew, Net, NetGlobal, Set2SetNet, SortPoolNet
from set2set import Set2Set
import torch.nn.functional as F
import argparse
import os
import pickle
import json
import copy
import random
import numpy as np
from torch.utils.data import random_split
from logger import config_logger
from tqdm import tqdm

def test(model,loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out,data.y,reduction='sum').item()
    return correct / len(loader.dataset),loss / len(loader.dataset)

def main(args, logger):
    for item in vars(args).items():
        logger.info('%s : %s', item[0], item[1])
    args.device = 'cpu'
    if torch.cuda.is_available():
        args.device = 'cuda:0'
    use_node_attr = False
    if args.dataset == 'FRANKENSTEIN' or args.dataset == 'ENZYMES':
        use_node_attr = True
    dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, use_node_attr=use_node_attr)
    if dataset.data.x is None:
        dataset.data.x = torch.ones(sum(dataset.data.num_nodes), 2)
        dataset.data.x[:, -1] = 0
        x_slices = []
        start = 0
        for num in dataset.data.num_nodes:
            x_slices.append(start)
            start = start + num
        x_slices.append(start)
        dataset.slices['x'] = torch.IntTensor(x_slices)
    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features


    # model = Net(args).to(args.device)

    # model.load_state_dict(torch.load('model.pth'))
    # with open('data.pickle', 'rb') as f:
    #     data = pickle.load(f)
    # data = data.to(args.device)
    # out = model(data)
    all_test_acc, all_test_loss, all_val_loss, all_val_acc = [], [], [], []
    Model = Net
    if args.model_type == 'MAGPool':
        Model = MAGPoolGCN
    elif args.model_type == 'MAGPool-global':
        Model = MAGPoolGCNGlobal
    elif args.model_type == 'SAG':
        Model = Net
    elif args.model_type == 'MAGPool-new':
        Model = MAGPoolGCNNew
    elif args.model_type == 'SAG-global':
        Model = NetGlobal
    elif args.model_type == 'Set2set':
        Model = Set2SetNet
    elif args.model_type == 'SortPool':
        Model = SortPoolNet
    else:
        raise Exception('Unrecognized model type %s', args.model_type)

    seed = args.seed
    fold_k = args.fold_k
    prefix = args.model_type
    if args.ablation is not None:
        prefix = prefix + '-' + args.ablation
    prefix = prefix + '_' + args.att_type + '_' + args.att_weight_type + '_' + args.score_type + '_' + \
             args.att_pooling_type + '_headnum' + str(args.head_num) + '_hid' + str(args.nhid) + '_lr' + str(args.lr) \
             + '_pr' + str(args.pooling_ratio) + '_dr' + str(args.dropout_ratio)
    '''Repeat the experiment with 20 different seeds'''
    for m in range(args.turn_num):
        subset_length = int(len(dataset) / fold_k)
        last_length = len(dataset) - subset_length * (fold_k - 1)
        subset_lengths = [subset_length] * (fold_k - 1) + [last_length]
        '''Using 10-fold validation method'''
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        seed += random.randrange(10000)
        cur_turn_test_acc, cur_turn_test_loss, cur_turn_val_acc, cur_turn_val_loss = [], [], [], []
        subsets = random_split(dataset, subset_lengths)
        for n in range(fold_k):
            cur_val_set = subsets[n]
            if n == 0:
                cur_training_set = copy.deepcopy(subsets[1])
            else:
                cur_training_set = copy.deepcopy(subsets[0])
            training_indices = []
            for i in range(fold_k):
                if i != n:
                    training_indices.extend(subsets[i].indices)
            cur_training_set.indices = training_indices

            train_loader = DataLoader(cur_training_set, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(cur_val_set, batch_size=args.batch_size, shuffle=False)
            # test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

            min_loss = 1e10
            max_val_acc = 0
            patience = 0

            model = Model(args).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            logger.info('\n')
            logger.info('****************************************')
            logger.info('\n')
            logger.info('Start experiment turn %d-%d', m, n)
            logger.info('\n')

            if args.model_type == 'SAG':
                model_path = os.path.join(args.dataset, args.model_type, '_num' + str(m) + '-' + str(n) + '_latest.pth')
            else:
                model_path = os.path.join(args.dataset, args.model_type,
                                          prefix + '_num' + str(m) + '-' + str(n) + '_latest.pth')

            if not os.path.exists(args.dataset):
                os.makedirs(args.dataset)
            if not os.path.exists(os.path.join(args.dataset, args.model_type)):
                os.makedirs(os.path.join(args.dataset, args.model_type))
            for epoch in range(args.epochs):
                model.train()
                for i, data in enumerate(train_loader):
                    data = data.to(args.device)
                    out = model(data)
                    loss = F.nll_loss(out, data.y)
                    logger.info("Training loss:%.5f", loss.item())
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                val_acc, val_loss = test(model, val_loader)
                logger.info("Validation loss:%.5f\taccuracy:%.5f", val_loss, val_acc)
                if val_loss < min_loss:
                    torch.save(model.state_dict(), model_path)
                    logger.info('=====================')
                    logger.info("Model saved at epoch %d", epoch)
                    logger.info('=====================')
                    min_loss = val_loss
                    max_val_acc = val_acc
                    patience = 0
                else:
                    patience += 1
                if patience > args.patience:
                    break

            model = Model(args).to(args.device)
            model.load_state_dict(torch.load(model_path))
            # test_acc, test_loss = test(model, test_loader)
            logger.info('\n')
            logger.info('++++++++++++++++++++++++++')
            logger.info('\n')
            logger.info('Best Val accuracy: %.5f, loss: %.5f', max_val_acc, min_loss)
            # logger.info("Test accuracy:%.5f", test_acc)
            # cur_turn_test_acc.append(test_acc)
            # cur_turn_test_loss.append(test_loss)
            cur_turn_val_acc.append(max_val_acc)
            cur_turn_val_loss.append(min_loss)
            logger.info('\n')
            logger.info('End experiment turn %d-%d', m, n)
            logger.info('\n')
            logger.info('****************************************')
            logger.info('\n')
        all_test_acc.append(cur_turn_test_acc)
        all_test_loss.append(cur_turn_test_loss)
        all_val_acc.append(cur_turn_val_acc)
        all_val_loss.append(cur_turn_val_loss)
    all_results = {'test_acc': all_test_acc, 'test_loss': all_test_loss, 'val_acc': all_val_acc,
                   'val_loss': all_val_loss}
    json_file_name = prefix + '_results.json'
    with open(os.path.join(args.dataset, json_file_name), 'w') as f:
        json.dump(all_results, f)
    logger.info('All experiment results saved!')
    acc_mean = np.mean(all_val_acc)
    acc_std = np.std(all_val_acc)
    return acc_mean, acc_std

def generate_permutation(all_parameters):
    res = []
    for i, parameter_set in enumerate(all_parameters):
        if i == 0:
            for parameter in parameter_set:
                res.append([parameter])
        else:
            res_copy = copy.deepcopy(res)
            for j, parameter in enumerate(parameter_set):
                if j == 0:
                    for sub_res in res:
                        sub_res.append(parameter)
                else:
                    for sub_res in res_copy:
                        res.append(sub_res + [parameter])
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=767,
                        help='seed')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay')
    parser.add_argument('--nhid', type=int, default=128,
                        help='hidden size')
    parser.add_argument('--pooling_ratio', type=float, default=0.5,
                        help='pooling ratio')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio')
    parser.add_argument('--dataset', type=str, default='DD',
                        help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
    parser.add_argument('--turn_num', type=int, default=20, help='The number of turns run for each 10-fold experiment')
    parser.add_argument('--epochs', type=int, default=100000,
                        help='maximum number of epochs')
    parser.add_argument('--patience', type=int, default=50,
                        help='patience for early stopping')
    parser.add_argument('--model_type', type=str, default='MAGPool-new',
                        help='MAGPool/MAGPool-global/MAGPool-new/SAG/SAG-global/Set2set')
    '''Attention type: 1)global: the hidden state will be weighted by the global attention score; 2)local: The sub 
    hidden state will be weighted by respect sub attention score'''
    parser.add_argument('--head_num', type=int, default=2, help='The head number for multihead model')
    parser.add_argument('--att_weight_type', type=str, default='global', help='global/local')
    '''The approach to obtain att score, 1)complex: the att score will be calcualted via Wx * x, 2)single: the att
    score is Wx'''
    parser.add_argument('--score_type', type=str, default='complex', help='single/complex')
    '''The attention type, 1)simple: will use a GCN to obtain the attention weight which will also be used to obtain
    the ranking score, 2)standard: a MLP will be used to obtain the attention weight while another GCN is used to 
    get the ranking score on the pooled attention-weighted representation'''
    parser.add_argument('--att_type', type=str, default='standard', help='simple/standard')
    parser.add_argument('--att_pooling_type', type=str, default='max', help='mean/max/min/concat')
    parser.add_argument('--fold_k', type=int, default=10, help='The validation fold size in training')
    parser.add_argument('--grid_search', action='store_true', help='Whether to execute the grid search to find out the '
                                                                 'bset parameters')
    parser.add_argument('--sort_k', type=int, default=14, help='Whether to execute the grid search to find out the '
                                                                   'bset parameters')
    parser.add_argument('--ablation', type=str, default=None, help='The variant type of model, 1)noatt means no '
            'self-attention and the results will be directly used for ranking, 2)noscore means no score GCN is applied'
            'and the results from attention will be directly used for ranking, 3)mlpatt: using MLP to calculate '
            'attention scores, 4)mlpscore: using mlp to calculate ranking scores, 5)nofilter: all nodes will be remained'
            '6)noreadout: only the results from the last layer is used to get the prediction')

    args = parser.parse_args()
    logger = config_logger(os.path.join(args.dataset, args.model_type))

    if args.grid_search:
        pooling_rates = [0.75, 0.5, 0.25]
        nhids = [64, 128, 256]
        head_nums = [2, 4]
        lrs = [0.001, 0.0005, 0.00025]
        parameter_permutations = generate_permutation([pooling_rates, nhids, head_nums, lrs])
        best_result, best_acc_mean, best_acc_std = 0, 0, 0
        best_parameter = []
        for parameter_permutation in tqdm(parameter_permutations):
            args.pooling_ratio, args.nhid, args.head_num, args.lr = parameter_permutation[0], \
                            parameter_permutation[1], parameter_permutation[2], parameter_permutation[3]
            cur_acc, cur_std = main(args, logger)
            cur_result = cur_acc + 0.2 * cur_std
            if cur_result > best_result:
                best_parameter = parameter_permutation
                best_acc_mean, best_acc_std = cur_acc, cur_std
        logger.info('\n')
        logger.info('==============================')
        logger.info('Best parameter is pooling rate: %.3f, hidden size: %d, head number: %d, learning rate: %.5f',
                    best_parameter[0], best_parameter[1], best_parameter[2], best_parameter[3])
        logger.info('Best mean acc is %.3f, best acc std is %.3f', best_acc_mean, best_acc_std)
    else:
        main(args, logger)
