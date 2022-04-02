# @Time    : 2021/12/21 11:12
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : main
# @Project Name :new_code

import tensorflow as tf
import argparse
from scipy.sparse.linalg import eigs
import numpy as np
import pickle
from utilsod.cal_graph import *
from data_loder.data_utils import *
from models.trainer import model_train
from models.tester import *

parser =argparse.ArgumentParser()
parser.add_argument('--n_zones', type=int, default=59)
parser.add_argument('--n_his', type=int, default=15)
parser.add_argument('--n_pred', type=int, default=9)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--save', type=int, default=10)
parser.add_argument('--k_s', type=int, default=3)
parser.add_argument('--k_t', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--path_file', type=str, default='./data_loder/datasets/train_data.pkl')
parser.add_argument('--path_file_adj', type=str, default='./data_loder/datasets/小区距离矩阵.xlsx')
parser.add_argument('--path_file_meta', type=str, default='./OD_NLSTM_model/model.ckpt-2850.meta')
parser.add_argument('--ckpt_name', type=str, default='OD_NLSTM_model')
# parser.add_argument('--inf_mode', type=str, default='merge')
args = parser.parse_args()
print(f'Training configs: {args}')

n_zones,n_his,n_pred = args.n_zones,args.n_his,args.n_pred
k_s,k_t =args.k_s,args.k_t
blocks = [[n_zones, 32, 64], [64, 32, 128]]

#load adjacency matrix w which saves the distance between each zones
w =weight_matrix (args.path_file_adj)
#normalization laplac
L = build_laplacttion(w)
#k_s-order Chebyshev polynomial approximation
Lk = cheb_ploy(L, k_s, n_zones)
tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))
#if you want you can change the function first_approx(W, n),as so called GCN.

#data preprocessing
path_file = args.path_file#training
# path_file = args.path_file = './data_loder/datasets/test_date.pkl'#for test
dataloder = Dataloder(path_file,n_his+n_pred,n_zones)
time_feature,o_datas_,d_datas_,o_stats,d_stats = dataloder.gen_data()

def main():
    model_train(o_datas_,d_datas_,blocks,args)
    # model_test(o_datas_,d_datas_,o_stats,args.batch_size,
    #            n_his,n_pred,args.path_file_meta,args.ckpt_name,single_day=True,one_day=1,peak=8)
if __name__ == '__main__':
    main()



