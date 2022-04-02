# @Time    : 2021/12/21 17:47
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : metrics_utils
# @Project Name :new_code

import numpy as np


# send a array with shape(batch_size,zones,zones)

def SMAPE(y, y_hat):
    y = np.reshape(y,(-1,))
    y_hat = np.reshape(y_hat, (-1,))
    index_1 = np.where(y>=5)
    return np.mean(np.abs(y[index_1] - y_hat[index_1]) / (abs(y[index_1])+abs(y_hat[index_1])))


def MSE(y, y_hat):
    y = np.reshape(y, (-1,))
    y_hat = np.reshape(y_hat, (-1,))
    index_1 = np.where(y >= 0)
    return (np.mean((y[index_1] - y_hat[index_1]) ** 2))


def MAE(y, y_hat):
    y = np.reshape(y, (-1,))
    y_hat = np.reshape(y_hat, (-1,))
    index_1 = np.where(y >= 0)
    return np.mean(np.abs(y[index_1] - y_hat[index_1]))


# send a tensor with the shape(n_day,n_hours,batch_size,zones,zones)
def evaluation(Y, Y_hat):
    each_day_earro = []
    o_day_erro = []
    d_day_erro = []
    for day in range(np.shape(Y)[0]):
        each_hours_erro = []
        o_hous_erro = []
        d_hours_erro = []
        for hours in range(np.shape(Y)[1]):
            each_hours_erro.append([SMAPE(Y[day][hours], Y_hat[day][hours]), MSE(Y[day][hours], Y_hat[day][hours]),
                                    MAE(Y[day][hours], Y_hat[day][hours])])
            Y_O = np.sum(Y[day][hours],axis=2)
            Y_hat_O = np.sum(Y_hat[day][hours], axis=2)
            Y_D = np.sum(Y[day][hours], axis=1)
            Y_hat_D = np.sum(Y_hat[day][hours], axis=1)
            o_hous_erro.append([SMAPE(Y_O,Y_hat_O),MSE(Y_O,Y_hat_O),MAE(Y_O,Y_hat_O)])
            d_hours_erro.append([SMAPE(Y_D, Y_hat_D), MSE(Y_D, Y_hat_D), MAE(Y_D, Y_hat_D)])
        each_day_earro.append(each_hours_erro)
        o_day_erro.append(o_hous_erro)
        d_day_erro.append(d_hours_erro)
    each_day_earro = np.array(each_day_earro)
    o_day_erro = np.array(o_day_erro)
    d_day_erro = np.array(d_day_erro)
    #OD
    print('For the {} step, The SMAPE is {}, The MSE is {}, and the MAE is {}'
          .format(0, np.mean(each_day_earro[:, 0, 0]), np.mean(each_day_earro[:, 0, 1]),
                  np.mean(each_day_earro[:, 0, 2])))

    print('For the {} step, The SMAPE is {}, The MSE is {}, and the MAE is {}'.
          format(1, np.mean(each_day_earro[:, 1, 0]), np.mean(each_day_earro[:, 1, 1]),
                 np.mean(each_day_earro[:, 1, 2])))

    print('For the {} step, The SMAPE is {}, The MSE is {}, and the MAE is {}'.
          format(2, np.mean(each_day_earro[:, 2, 0]), np.mean(each_day_earro[:, 2, 1]),
                 np.mean(each_day_earro[:, 2, 2])))
    #O
    print('For the {} step, The SMAPE is {}, The MSE is {}, and the MAE is {} for Origin'
          .format(0, np.mean(o_day_erro[:, 0, 0]), np.mean(o_day_erro[:, 0, 1]),
                  np.mean(o_day_erro[:, 0, 2])))
    print('For the {} step, The SMAPE is {}, The MSE is {}, and the MAE is {} for Origin'.
          format(1, np.mean(o_day_erro[:, 1, 0]), np.mean(o_day_erro[:, 1, 1]),
                 np.mean(o_day_erro[:, 1, 2])))

    print('For the {} step, The SMAPE is {}, The MSE is {}, and the MAE is {} for Origin'.
          format(2, np.mean(o_day_erro[:, 2, 0]), np.mean(o_day_erro[:, 2, 1]),
                 np.mean(o_day_erro[:, 2, 2])))
    #D
    print('For the {} step, The SMAPE is {}, The MSE is {}, and the MAE is {} for Destination'
          .format(0, np.mean(d_day_erro[:, 0, 0]), np.mean(d_day_erro[:, 0, 1]),
                  np.mean(d_day_erro[:, 0, 2])))
    print('For the {} step, The SMAPE is {}, The MSE is {}, and the MAE is {} for Destination'.
          format(1, np.mean(d_day_erro[:, 1, 0]), np.mean(d_day_erro[:, 1, 1]),
                 np.mean(d_day_erro[:, 1, 2])))

    print('For the {} step, The SMAPE is {}, The MSE is {}, and the MAE is {} for Destination'.
          format(2, np.mean(d_day_erro[:, 2, 0]), np.mean(d_day_erro[:, 2, 1]),
                 np.mean(d_day_erro[:, 2, 2])))



