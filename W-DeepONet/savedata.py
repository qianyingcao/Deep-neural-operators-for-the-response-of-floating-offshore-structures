# import tensorflow.compat.v1 as tf
import tensorflow as tf
import numpy as np
import sys
from fnn import FNN
import os
import matplotlib.pyplot as plt
import scipy
plot_folder = './Plot/'    
class SaveData:
    def __init__(self):
        pass

    def save(self, sess, fnn_model, W_T1, b_T1,  W_B1, b_B1,  X1min, X1max,  f1_ph,  u1_ph, u2_ph, x1_ph, data, num_test, save_results_to,p1,num):
        
        x1_test, f1_test, u1_test, u2_test = data.testbatch(num_test)
        
        test_dict = {f1_ph: f1_test, u1_ph: u1_test, x1_ph: x1_test, u2_ph: u2_test}
        
        u1_T = fnn_model.fnn_T1(W_T1, b_T1, x1_ph, X1min, X1max)
        u1_B = fnn_model.fnn_B1(W_B1, b_B1, f1_ph)

        
        u1_pred = tf.einsum('ijk, lk->il', u1_B[:,:,0:p1//2], u1_T[:,0:p1//2])
        u2_pred = tf.einsum('ijk, lk->il', u1_B[:,:,p1//2:p1], u1_T[:,p1//2:p1])
        u1_pred = tf.reshape(u1_pred,(-1,num,1))
        u2_pred = tf.reshape(u2_pred,(-1,num,1))

        u1_pred_, u2_pred_  = sess.run([u1_pred,u2_pred], feed_dict=test_dict)  
        
        u1_test, u2_test = data.decoder(u1_test, u2_test)
        u1_pred_, u2_pred_ = data.decoder(u1_pred_, u2_pred_) 
        f1_test = np.reshape(f1_test, (f1_test.shape[0], -1))

        u1_pred_ = np.reshape(u1_pred_[:,:,0:1], (u1_test[:,:,0:1].shape[0], u1_test[:,:,0:1].shape[1]))
        u2_pred_ = np.reshape(u2_pred_[:,:,0:1], (u2_test[:,:,0:1].shape[0], u2_test[:,:,0:1].shape[1]))
        x1_test = np.reshape(x1_test[:,0:1],(u1_test[:,:,0:1].shape[1]))
        u1_test = np.reshape(u1_test[:,:,0:1], (u1_test[:,:,0:1].shape[0], u1_test[:,:,0:1].shape[1]))
        u2_test = np.reshape(u2_test[:,:,0:1], (u2_test[:,:,0:1].shape[0], u2_test[:,:,0:1].shape[1]))

        err_u1 = np.mean(np.square(u1_pred_ - u1_test))
        err_u2 = np.mean(np.square(u2_pred_ - u2_test))
        
        print('MSE Error_u1: %.3f'%(err_u1))
        print('MSE Error_u2: %.3f'%(err_u2))

        err_u1 = np.reshape(err_u1, (-1, 1))
        err_u2 = np.reshape(err_u2, (-1, 1))
        
        np.savetxt(save_results_to+'/err_u1.txt', err_u1, fmt='%e')
        np.savetxt(save_results_to+'/err_u2.txt', err_u2, fmt='%e')
        
        
        scipy.io.savemat(save_results_to+'Signals_pred_DeepONet.mat', 
                     mdict={ 'u1_err': err_u1, 'u2_err': err_u2,
                             'x1_test': x1_test, 
                            'u1_test': u1_test, 'u2_test': u2_test,
                            'u1_pred': u1_pred_, 'u2_pred': u2_pred_})