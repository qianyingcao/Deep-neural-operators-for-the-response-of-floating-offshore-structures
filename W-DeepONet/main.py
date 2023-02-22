import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from dataset import DataSet
from fnn import FNN
from savedata import SaveData
import scipy.io as io
from scipy.fft import fft, fftfreq, fftshift

np.random.seed(1234)
#tf.set_random_seed(1234)
# Case1: 
# Case2: 
# Case3: 
# Case4: 
# Case5: 
# Case6: 
# Case 7: 
# Case 8: 
#output dimension of Branch/Trunk
num_train = 14400
num_vali = 100
num_test = 100
p1 = 300
num = 500
num_trunk1 = 1
#branch net
layer_B1 = [num*2, 128, 128, 128, p1]
#trunk net
layer_T1 = [num_trunk1, 128, 128, p1]
#resolution
h = num
#batch_size
bs = 1200
bs_vali = 50
bs_time = 500
epochs = 1000000
lbfgs_iter = 5000
beta1 = 0.0001
save_index = 3
current_directory = os.getcwd()    
case = "Case_"
folder_index = str(save_index)
results_dir = "/" + case + folder_index +"/"
save_results_to = current_directory + results_dir
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)  
    
def main():
    data = DataSet(num, bs,bs_time)
    x1_train, f1_train,  u1_train,u2_train, X1min, X1max = data.minibatch()

    f1_ph = tf.placeholder(shape=[None, 1, num*2], dtype=tf.float32) #[bs, f_dim]
    u1_ph = tf.placeholder(shape=[None, num, 1], dtype=tf.float32) #[bs, x_num, 1]
    u2_ph = tf.placeholder(shape=[None, num, 1], dtype=tf.float32) #[bs, x_num, 1]
    x1_ph = tf.placeholder(shape=[num, num_trunk1], dtype=tf.float32)

    learning_rate1 = tf.placeholder(tf.float32, shape=[])
    fnn_model = FNN()
    
    # Branch net
    W_B1, b_B1 = fnn_model.hyper_initial(layer_B1)
    u1_B = fnn_model.fnn_B1(W_B1, b_B1, f1_ph)

    #Trunk net
    W_T1, b_T1 = fnn_model.hyper_initial(layer_T1)
    u1_T = fnn_model.fnn_T1(W_T1, b_T1, x1_ph, X1min, X1max) 

    u1_pred, u2_pred = fnn_model.fnn_net(u1_B,u1_T,p1,num)
    # u1_pred =u_pred[:,:,0:p1//2]
    # u2_pred =u_pred[:,:,p1//2:p1]
    
    regularizers1 = fnn_model.l2_regularizer(W_B1)
    loss1 = tf.reduce_mean(tf.square(u1_ph - u1_pred)) + tf.reduce_mean(tf.square(u2_ph - u2_pred)) + beta1*regularizers1

    lbfgs_buffer1 = []
    train1_adam = tf.train.AdamOptimizer(learning_rate = learning_rate1).minimize(loss1)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))  
    sess.run(tf.global_variables_initializer())
    
    def callback1(lbfgs_buffer,loss):
        lbfgs_buffer = np.append(lbfgs_buffer, loss)
        print(loss)
        
    n = 0
    start_time = time.perf_counter()
    time_step_0 = time.perf_counter()
    
    train_loss = np.zeros((epochs+1, 2))
    test_loss = np.zeros((epochs+1, 2))    
    while n <= epochs:
        
        if n <50000:
                lr1 = 0.0005
        elif (n < 80000):
                lr1 = 0.0005
        elif (n < 200000):
                lr1 = 0.00001
        else:
                lr1 = 0.00001

        x1_train,  f1_train,  u1_train, u2_train, _, _ = data.minibatch()

        train_dict = {f1_ph: f1_train, u1_ph: u1_train, u2_ph: u2_train, x1_ph: x1_train, learning_rate1: lr1}
        loss1_,  _ = sess.run([loss1,train1_adam], feed_dict = train_dict)   

        if n%1 == 0:
            x1_vali, f1_vali, u1_vali, u2_vali= data.valibatch(bs_vali)
            u1_vali_,u2_vali_ = sess.run([u1_pred, u2_pred],feed_dict={f1_ph: f1_vali, x1_ph: x1_vali})
            test_mse_u1 = np.mean(np.square(u1_vali_ - u1_vali))+np.mean(np.square(u2_vali_ - u2_vali))
            time_step_1000 = time.perf_counter()
            T = time_step_1000 - time_step_0
            time_step_0 = time.perf_counter()
        if  n%1000 == 0:
            print('Epoch: %d, Training Loss1: %.3e, Testing U1 : %.3e,  Time (secs): %.3f'%(n, loss1_, test_mse_u1, T))
        train_loss[n,0] = loss1_
        test_loss[n,0] = test_mse_u1
        n += 1
    
    x1_train,  f1_train,  u1_train,u2_train, _, _= data.alldata()

    # optimizer1 = tf.contrib.opt.ScipyOptimizerInterface(loss1,
    #                                                   method='L-BFGS-B',
    #                                                   options={'maxiter': lbfgs_iter,
    #                                                             'maxfun': lbfgs_iter,
    #                                                             'maxcor': 100,
    #                                                               'maxls': 50,
    #                                                               'ftol': 1.0 * np.finfo(float).eps})    

    # train_dict = {f1_ph: f1_train, u1_ph: u1_train,u2_ph: u2_train, x1_ph: x1_train, learning_rate1: lr1}
    # optimizer1.minimize(sess, feed_dict=train_dict, fetches=[lbfgs_buffer1, loss1],loss_callback= callback1)
    
    save_models_to = save_results_to +"model/"
    if not os.path.exists(save_models_to):
        os.makedirs(save_models_to)      

    stop_time = time.perf_counter()
    print('Elapsed time (secs): %.3f'%(stop_time - start_time))
    num_epoch = train_loss.shape[0]
    x = np.linspace(0, num_epoch-1, num_epoch)
    np.savetxt(save_results_to+'/epoch.txt', x)       
    np.savetxt(save_results_to+'/train_loss.txt', train_loss)
    np.savetxt(save_results_to+'/test_loss.txt', test_loss)

    data_save = SaveData()

    data_save.save(sess, fnn_model, W_T1, b_T1, W_B1, b_B1, X1min, X1max,  f1_ph, u1_ph,u2_ph, x1_ph, data, num_test, save_results_to,p1,num)
    
    ## Plotting the loss history
   
    fig = plt.figure(constrained_layout=False, figsize=(10, 6))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(x, train_loss[:,0], color='blue', label='Training Loss')
    ax.plot(x, test_loss[:,0], color='red', label='Testing Loss')
    ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    ax.legend(loc='upper left')
    fig.savefig(save_results_to+'loss_his_u1.png')
    plt.show()


if __name__ == "__main__":
    main()
