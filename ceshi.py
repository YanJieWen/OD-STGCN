# @Time    : 2021/12/21 11:19
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : ceshi
# @Project Name :new_code
import numpy as np

from data_loder.data_utils import *
import tensorflow as tf
from models.base_model import *
import pandas as pd
from scipy.sparse.linalg import eigs
from data_loder.data_utils import *
# #加载邻接矩阵,
# w = pd.read_excel('./data_loder/datasets/小区距离矩阵.xlsx',header=None).values
# if set(np.unique(w))=={0,1}:#对邻接矩阵进行转换，进行距离转换
#     print('The input graph is a 0/1 matrix; set "scaling" to False.')
#     scaling = False
# else:
#     n=w.shape[0]
#     w=w/10000.
#     w2,w_mask =w*w,np.ones([n,n])-np.identity(n)#np.identify创建一个对角线为1的方阵
#     w = np.exp(-w2 / 0.1) * (np.exp(-w2 / 0.1) >= 0.5) * w_mask#*为元素乘,算法思想的融入
# #获得拉普拉斯矩阵
# d=np.sum(w,axis=1)
# l=-w
# l[np.diag_indices_from(l)]=d#将d转为对角D矩阵
# for i in range(n):
#     for j in range(n):
#         if (d[i] > 0) and (d[j] > 0):
#             l[i, j] = l[i, j] / np.sqrt(d[i] * d[j])
# lambda_max = eigs(l, k=1, which='LR')[0][0].real
# l=np.mat(2 * l / lambda_max - np.identity(n))#mat可以从字符串生成数组，而array只能从列表中生成
# L0, L1 = np.mat(np.identity(n)), np.mat(np.copy(l))
# if 3> 1:
#     L_list = [np.copy(L0), np.copy(L1)]
#     for i in range(3 - 2):
#         Ln = np.mat(2 * l * L1 - L0)
#         L_list.append(np.copy(Ln))
#         L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Ln))
#     Lk=np.concatenate(L_list, axis=-1)
# tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))
# blocks = [[59,32,64],[64,32,128]]
# x_O = tf.placeholder(tf.float32,[None,15+1,59,59],name='input_original')
# x_D = tf.placeholder(tf.float32, [None, 15+1,59,59], name='input_destination')
# time_feature = tf.placeholder(tf.int32,[None,59+1,10])
# keep_prob = tf.placeholder(tf.float32, name='keep_prob')
# train_loss,single_pred= intergrate_od_output(x_O,x_D, 15, 3, 3, blocks, keep_prob)
# print(train_loss)

test_data = Dataloder('./data_loder/datasets/test_date.pkl', 24, 59)
time_feature, o_datas_, d_datas_, o_stats, d_stats = test_data .gen_data()#生成(seq_len,24,59,59)大小的数据
saver = tf.train.import_meta_graph("./OD_model/model.ckpt-2850.meta")
print(o_stats,d_stats)
#随机选取一天的数据
o_oneday = np.array(o_datas_)[0,0:16,:,:]
o_oneday_ = np.expand_dims(o_oneday,0)
d_oneday = np.array(d_datas_)[0,0:16,:,:]
d_oneday_ = np.expand_dims(d_oneday,0)
with tf.Session() as sess:
    graph = tf.get_default_graph()
    saver.restore(sess, tf.train.latest_checkpoint('OD_model'))
    pred = graph.get_collection('y_pred')
    pred_ =sess.run(pred,feed_dict={'input_original:0':o_oneday_,'input_destination:0':d_oneday_,'keep_prob:0':1.0})
    pred_ = np.array(np.squeeze(pred_))
    new_pred = pred_*(o_stats['O_max']-o_stats['O_min'])+o_stats['O_min']
    predict = pd.DataFrame(new_pred)
    predict.to_csv('./predict.csv')
    label_ = o_oneday_[:,-1,:,:]
    new_label = label_*(o_stats['O_max']-o_stats['O_min'])+o_stats['O_min']
    label = pd.DataFrame(np.squeeze(new_label))
    label.to_csv('./label.csv')

