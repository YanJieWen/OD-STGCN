# @Time    : 2021/12/21 16:19
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : tester
# @Project Name :new_code

import numpy as np

from data_loder.data_utils import *
import tensorflow as tf
from models.base_model import *
from data_loder.data_utils import *
import time
from utilsod.metrics_utils import *
import pandas as pd
import random

random.seed(3)
# test_data = Dataloder()
# time_feature, o_datas_, d_datas_, o_stats, d_stats = test_data.gen_data()  # 生成(seq_len,24,59,59)大小的数据
# n_his, n_pred = args.n_his, args.n_pred
# n_frame = n_his + n_pred
# n_zones = args.n_zones
# data_path_file = args.path_file

def model_test(o_datas_,d_datas_,o_stats,batch_size,n_his,n_pred,meta_file,ckpt_name,single_day=True,one_day=1,peak=8):

    saver = tf.train.import_meta_graph(meta_file)
    test_sess = tf.Session()
    graph = tf.get_default_graph()
    saver.restore(test_sess, tf.train.latest_checkpoint(ckpt_name))
    pred = graph.get_collection('y_pred')
    if single_day ==False:
        step_idx  = np.arange(3, n_pred + 1, 3) - 1#to generate the muti time steps
        # step_idx = np.array([0,1,2])
        y_predict,y_label = muti_pred(test_sess, pred, o_datas_,d_datas_,o_stats,batch_size, n_his, n_pred, step_idx)
        evaluation(y_label,y_predict)
    else:#output the one day and save to the csv
    #choose one day,five days in
        one_durio_to_csv(test_sess,pred,n_his,o_datas_,d_datas_,o_stats,one_day,peak)
    # return y_predict

def muti_pred(test_sess, pred, o_datas_,d_datas_,o_stats,batch_size, n_his, n_pred, step_idx):
    pred_list = []
    label_list = []
    for (o_,d_) in gen_batch(o_datas_,d_datas_,batch_size,dynamic_batch=True, shuffle=True):#ensure the same idx
        test_o_seq = np.copy(o_[:, 0:n_his + 1, :, :])
        test_d_seq = np.copy(d_[:, 0:n_his + 1, :, :])
        step_list = []
        step_list_label = []
        if np.shape(test_o_seq)[0]==batch_size:#to ensure the same as the batchsize
            for j in range(n_pred):
                st_time = time.time()
                pred_y = test_sess.run(pred,
                                feed_dict={'input_original:0':test_o_seq,'input_destination:0':test_d_seq,'keep_prob:0':1.0})
                end_time = time.time()
                print('your model test used {} for each batch'.format(end_time-st_time))
                if isinstance(pred_y , list):
                    pred_y = np.array(pred_y [0])
                test_o_seq = recurrence_seq(test_o_seq,n_his,pred_y )
                test_d_seq = recurrence_seq(test_d_seq,n_his,pred_y )
                if j in step_idx:
                    # label_y = inverse_pred(test_o_seq[:,n_his:n_his+1,:,:],o_stats)#the erro label
                    # np.copy(o_[:,j+n_his,:,:])
                    label_y = inverse_pred(np.copy(o_[:,j+n_his,:,:]), o_stats)
                    pred_y = inverse_pred(pred_y,o_stats)
                    step_list.append(pred_y)
                    step_list_label.append(label_y)
        #  pred_list -> [n_day, [3,6,9], n_zones,n_zones),the same as the label_list
            pred_list.append(step_list)
            label_list.append(step_list_label)
    return pred_list,label_list
def recurrence_seq(test_seq,n_his,pred_y ):#to get updated seq which replaced by pred located time[-1]
    test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
    test_seq[:, n_his - 1, :, :] = pred_y#the last number
    return test_seq
def inverse_pred(pred_y,o_stats):
    return pred_y*(o_stats['O_max']-o_stats['O_min'])+o_stats['O_min']

def one_durio_to_csv(test_sess,pred,n_his,o_datas_,d_datas_,o_stats,one_day=1,peak=8):#muti->one
    o_oneday = np.array(o_datas_)[24 * one_day + peak - n_his+1, 0:16, :, :]
    o_oneday_ = np.expand_dims(o_oneday, 0)
    d_oneday = np.array(d_datas_)[24 * one_day + peak -n_his+1, 0:16, :, :]
    d_oneday_ = np.expand_dims(d_oneday, 0)
    pred_ = test_sess.run(pred, feed_dict={'input_original:0': o_oneday_, 'input_destination:0': d_oneday_,
                                           'keep_prob:0': 1.0})
    pred_ = np.array(np.squeeze(pred_))
    new_pred = inverse_pred(pred_, o_stats)
    new_label = inverse_pred(o_oneday_[:, -1, :, :], o_stats)
    predict = pd.DataFrame(new_pred)
    predict.to_csv('./predict.csv')
    label = pd.DataFrame(np.squeeze(new_label))
    label.to_csv('./label.csv')




