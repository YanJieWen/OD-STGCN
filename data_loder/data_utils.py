# @Time    : 2021/12/21 11:16
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : data_utils
# @Project Name :new_code

import pickle
import numpy as np


class Dataloder():
    def __init__(self, path_file, n_frame, n_zone):
        self.path_file = path_file
        file = open(self.path_file, 'rb')
        self.datas = pickle.load(file)
        self.n_frame = n_frame
        self.n_zone = n_zone

    def gen_data(self):  # generate data of the  origin and the destination
        o_data = []
        d_data = []
        for k, v in self.datas.items():
            o_data.append([v[0], v[1]])
            d_data.append([v[0].transpose(), v[1]])
        # generate data sequence with the shape of(seq_len,n_frame,n_zone,n_zone) and time feature with the
        # shape (seq_len,n_frame,n_timef_feature)
        o_data = whether_ndarry(o_data)
        d_data = whether_ndarry(d_data)
        o_datas = [[o_data[i:i + self.n_frame][:, 0][j] for j in range(self.n_frame)]
                   for i in range(o_data.shape[0] - self.n_frame)]

        d_datas = [[d_data[i:i + self.n_frame][:, 0][j] for j in range(self.n_frame)]
                   for i in range(d_data.shape[0] - self.n_frame)]
        time_feature = [[o_data[i:i + self.n_frame][:, 1][j] for j in range(self.n_frame)]
                        for i in range(o_data.shape[0] - self.n_frame)]
        # the standardized parameters(the method is maxmin)
        o_stats = {'O_max': np.max(o_datas), 'O_min': np.min(o_datas)}
        d_stats = {'D_max': np.max(d_datas), 'D_min': np.min(d_datas)}
        o_datas_ = maxminmethod(o_datas, o_stats['O_max'], o_stats['O_min'])
        d_datas_ = maxminmethod(d_datas, d_stats['D_max'], d_stats['D_min'])
        return time_feature, o_datas_, d_datas_, o_stats, d_stats


# some samll functions
def whether_ndarry(a):
    if isinstance(a, list):
        return np.array(a)


def maxminmethod(a, max, min):
    return (a - min) / (max - min)


def gen_batch(inputs_O, inputs_D, batch_size, dynamic_batch=False, shuffle=False):  # generate data_batch
    len_inputs = len(inputs_O)
    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)
    for start_idx in range(0, len_inputs, batch_size):  # Ensure that the data is traversed
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:
                end_idx = len_inputs
            else:
                break
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)

        yield inputs_O[slide], inputs_D[slide]
