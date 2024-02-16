import pyabcore
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import os
from time import time
import torch
import networkx as nx
from collections import Counter
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data.dataset import Dataset
root_path, _ = os.path.split(os.path.abspath(__file__)) 

def get_labeled_node(df):

    pos = df[df[:,-1] == 1]
    neg = df[df[:,-1] == 0]

    total_u = list(set(df[:,0].numpy()))
    total_u.sort()
    pos_u = set(pos[:,0].numpy())

    neg_u = set(neg[:,0].numpy()) - pos_u

    pos_u_mask = torch.BoolTensor([(i in pos_u) for i in total_u])
    neg_u_mask = torch.BoolTensor([(i in neg_u) for i in total_u])
    u_l = (torch.zeros(len(total_u))-1).double()
    u_l = torch.where(neg_u_mask, 0., u_l)
    u_l = torch.where(pos_u_mask, 1., u_l)
    
    return u_l

def get_data():

    train_data = pd.read_csv(root_path+'/dataset/train.txt', names=['u', 'i', 'l'], delimiter='\t', dtype=int)
    test_data = pd.read_csv(root_path+'/dataset/test.txt', names=['u', 'i','l'], delimiter='\t', dtype=int)

    df = pd.concat((train_data, test_data))

    dataset = Dataset()

    dataset.max_u = max(df['u'])+1
    dataset.max_i = max(df['i'])+1

    df_labels = df[df['l'] != -1]
    
    dataset.all_edge = np.array(df[['u', 'i']], dtype=np.int32)

    dataset.train_edge = torch.LongTensor(np.array(train_data[['u', 'i', 'l']]))

    dataset.test_edge = torch.LongTensor(np.array(test_data[['u', 'i', 'l']]))
    
    dataset.train_u = list(set(train_data['u']))
    dataset.train_u.sort()
    dataset.train_u = torch.LongTensor(dataset.train_u)

    dataset.u_x = torch.FloatTensor(np.load(root_path+'/dataset/bdt_u_features.npy'))
    dataset.i_x = torch.FloatTensor(np.load(root_path+'/dataset/bdt_i_features.npy'))
    
    return dataset

def get_abcore(dataset, device, time_num):
    abcore = pyabcore.Pyabcore(dataset.max_u, dataset.max_i)
    abcore.index(dataset.all_edge)
    # print('finished, time:{}'.format(index_time - start_time))
    a = 2
    b = 1
    dataset.core_u_x = torch.BoolTensor([])
    dataset.core_i_x = torch.BoolTensor([])
    while 1:
        abcore.query(a, b)
        result_u = torch.BoolTensor(abcore.get_left())
        result_i = torch.BoolTensor(abcore.get_right())
        if(result_i.sum() < len(result_i)*0.01):
            print('max b:{}'.format(b-1))
            dataset.max_b = b-1
            break

        dataset.core_u_x = torch.cat((dataset.core_u_x, result_u.unsqueeze(-1)),dim=1)
        dataset.core_i_x = torch.cat((dataset.core_i_x, result_i.unsqueeze(-1)),dim=1)
        b += 1

    
    tmp_edge = dataset.test_edge.clone()
    tmp_edge[:,-1] = -1
    dataset.train_u_l = get_labeled_node(torch.cat((dataset.train_edge, tmp_edge)))

    tmp_edge = dataset.train_edge.clone()
    tmp_edge[:,-1] = -1

    dataset.test_u_l = get_labeled_node(torch.cat((tmp_edge, dataset.test_edge)))

    tmp_label = torch.where(dataset.test_u_l == -1, -1., dataset.train_u_l)
    dataset.test_u_l = torch.where((tmp_label == 1) * (dataset.test_u_l == 0), 1., dataset.test_u_l)

    dataset.train_u_l = dataset.train_u_l.long()
    dataset.test_u_l = dataset.test_u_l.long()

    dataset.test_edge = torch.cat((tmp_edge, dataset.test_edge))
    
    dataset.u_x = torch.cat((dataset.core_u_x, dataset.u_x),dim=1)
    dataset.i_x = torch.cat((dataset.core_i_x,dataset.i_x),dim=1)


    user_time = pd.read_csv(root_path+'/dataset/user_time.txt', names=['u', 't'], delimiter='\t', dtype=str)
    user_time['u'] = user_time['u'].astype(int)

    user_time['t'] = user_time['t'].str.split(',').apply(lambda x: [int(i) for i in x])

    times = user_time['t'].explode()
    dataset.time_min = times.min()
    dataset.time_max = times.max()

    time_length = int(np.ceil((dataset.time_max - dataset.time_min) / time_num))
    dataset.time_num = time_num

    time_stamps = times.astype(int)
    user_indices = user_time.index.repeat(user_time['t'].str.len())

    time_tensor = torch.tensor(time_stamps.values).float()
    user_idx = torch.tensor(user_indices.values)
    attacker_flags = dataset.train_u_l[user_idx]

    segment_indices = (time_tensor - dataset.time_min) // time_length
    if segment_indices.max() >= time_num:
        segment_indices[segment_indices == segment_indices.max()] = segment_indices.max() - 1

    assert segment_indices.max() < time_num


    attack_weights = torch.zeros(time_num)
    attack_time_centers = torch.zeros(time_num)

    for i in range(time_num):
        mask = (segment_indices == i) & (attacker_flags == 1)
        attack_weights[i] = mask.sum()
        assert attack_weights[i] > 0
        attack_time_centers[i] = time_tensor[mask].mean()

    attack_weights = attack_weights / attack_weights.max()

    max_distance = dataset.time_max - dataset.time_min
    scale_factor = 10 / max_distance
    user_time_weights = torch.zeros(dataset.max_u, time_num)

    att_target_dist = torch.abs(attack_time_centers[:, None] - attack_time_centers) * scale_factor
    sigma = 1.0
    att_target_gaussian_distances = torch.exp(-torch.pow(att_target_dist, 2) / (2 * sigma**2))
    att_target = (att_target_gaussian_distances * attack_weights)
    att_target = (att_target / att_target.max(dim=-1, keepdim=True).values).sum(dim=0)
    att_target = att_target / att_target.max()
    for idx in trange(dataset.max_u):
        user_timestamps = time_tensor[user_idx == idx].float()

        distances = torch.abs(user_timestamps[:, None] - attack_time_centers) * scale_factor

        gaussian_distances = torch.exp(-torch.pow(distances, 2) / (2 * sigma**2))

        weighted_distances = (gaussian_distances * attack_weights)


        user_time_weight = weighted_distances / weighted_distances.max(dim=-1, keepdim=True).values
        
        attention_weights = torch.nn.functional.softmax(torch.matmul(user_time_weight, att_target.T).squeeze(), dim=-1)
        user_time_weight = (attention_weights.unsqueeze(-1) * user_time_weight).sum(dim=0)
        user_time_weights[idx] = user_time_weight / user_time_weight.max()


    dataset.u_x = torch.cat((dataset.u_x, user_time_weights), dim=1)
    dataset.train_edge_x = torch.cat((dataset.u_x[dataset.train_edge[:,0]][:,:dataset.max_b],dataset.i_x[dataset.train_edge[:,1]][:,:dataset.max_b],dataset.u_x[dataset.train_edge[:,0]][:,dataset.max_b:],dataset.i_x[dataset.train_edge[:,1]][:,dataset.max_b:]),dim=1)

    dataset.test_edge_x = torch.cat((dataset.u_x[dataset.test_edge[:,0]][:,:dataset.max_b],dataset.i_x[dataset.test_edge[:,1]][:,:dataset.max_b],dataset.u_x[dataset.test_edge[:,0]][:,dataset.max_b:],dataset.i_x[dataset.test_edge[:,1]][:,dataset.max_b:]),dim=1)

    dataset.train_edge_y = dataset.train_edge[:,2]
    dataset.train_edge = dataset.train_edge[:,:2].t()
    dataset.train_edge[1] = dataset.train_edge[1] + dataset.max_u

    dataset.test_edge_y = dataset.test_edge[:,2]
    dataset.test_edge = dataset.test_edge[:,:2].t()
    dataset.test_edge[1] = dataset.test_edge[1] + dataset.max_u

    dataset.user_time_weights = user_time_weights

    dataset.user_time_weights = dataset.user_time_weights.to(device)
    dataset.train_u_l = dataset.train_u_l.long().to(device)
    dataset.test_u_l = dataset.test_u_l.long().to(device)
    dataset.u_x = dataset.u_x.to(device)
    dataset.i_x = dataset.i_x.to(device)

    dataset.train_edge_x = dataset.train_edge_x.to(device)

    dataset.test_edge_x = dataset.test_edge_x.to(device)

    dataset.train_edge_y = dataset.train_edge_y.to(device)
    dataset.train_edge = dataset.train_edge.to(device)

    dataset.test_edge_y = dataset.test_edge_y.to(device)
    dataset.test_edge = dataset.test_edge.to(device)

    return dataset


def get_abcore_data(device, time_num=8):
    dataset= get_data()
    dataset = get_abcore(dataset, device, time_num)

    return dataset
    
if __name__ == '__main__':
    get_abcore_data('cpu')
