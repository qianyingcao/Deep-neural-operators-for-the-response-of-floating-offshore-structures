# import tensorflow.compat.v1 as tf
import tensorflow as tf
import numpy as np
import scipy.io as io
import scipy
import sys
np.random.seed(1234)

class DataSet:
    def __init__(self, num, bs,bs_time):
        self.num = num
        self.bs = bs
        self.bs_time= bs_time
        self.F_train, self.U1_train,self.F_vali, self.U1_vali, self.F_test, self.U1_test, \
        self.X1_train, self.X1_vali, self.X1_test,  self.U2_train, \
        self.U2_vali, self.U2_test,\
            self.u1_mean, self.u1_std, self.u2_mean, self.u2_std = self.load_data(num)
            
    def decoder(self, x1, x2):
        
        x1 = x1*(self.u1_std + 1.0e-9) + self.u1_mean
        # x1 = x1*(self.u1_max - self.u1_min + 1.0e-9) + self.u1_min
        x2 = x2*(self.u2_std + 1.0e-9) + self.u2_mean
        x2 = x2*1e-3
        return x1, x2

    def load_data(self,num):

        data = io.loadmat('./Data/data.mat')
        
        #xx = np.linspace(0,1,num).reshape(-1,1)

        
        f1_train = data['f_train_A']
        f2_train = data['f_train_D']*1e2
        u1_train = data['u_train_A']
        u2_train = data['u_train_D']*1e3
        x1_train = data['x_train_A']
  
        f1_vali = data['f_vali_A']
        f2_vali = data['f_vali_D']*1e2
        u1_vali = data['u_vali_A']
        u2_vali = data['u_vali_D']*1e3
        x1_vali = data['x_vali_A']
     
        f1_test = data['f_test_A']
        f2_test = data['f_test_D']*1e2
        u1_test = data['u_test_A']
        u2_test = data['u_test_D']*1e3
        x1_test = data['x_test_A']
      
        
        
        f1_train_min = np.min(f1_train, 0)
        f1_train_max = np.max(f1_train, 0)
        f1_train_min = np.reshape(f1_train_min, (-1, 1, num))
        f1_train_max = np.reshape(f1_train_max, (-1, 1, num))
        F1_train = np.reshape(f1_train, (-1, 1, num))
        F1_vali = np.reshape(f1_vali, (-1, 1, num))
        F1_test = np.reshape(f1_test, (-1, 1, num))
        F1_train=(F1_train-f1_train_min)/(f1_train_max-f1_train_min+1.0e-9)
        F1_vali=(F1_vali-f1_train_min)/(f1_train_max-f1_train_min+1.0e-9)
        F1_test=(F1_test-f1_train_min)/(f1_train_max-f1_train_min+1.0e-9)
        
        

        f2_train_mean = np.mean(f2_train,0)
        f2_train_std = np.std(f2_train,0)
        F2_train = np.reshape(f2_train, (-1, 1, num))
        F2_vali = np.reshape(f2_vali, (-1, 1, num))
        F2_test = np.reshape(f2_test, (-1, 1, num))
        F2_train=(F2_train-f2_train_mean)/f2_train_std
        F2_vali=(F2_vali-f2_train_mean)/f2_train_std
        F2_test=(F2_test-f2_train_mean)/f2_train_std       

        F_train=np.concatenate((F1_train, F2_train), axis=2)
        F_vali=np.concatenate((F1_vali, F2_vali), axis=2)
        F_test=np.concatenate((F1_test, F2_test), axis=2)
        
        u1_train_mean = np.mean(u1_train,0)
        u1_train_std = np.std(u1_train,0)

        u1_train_mean = np.reshape(u1_train_mean, (-1, num, 1))
        u1_train_std = np.reshape(u1_train_std, (-1, num, 1))
        U1_train = np.reshape(u1_train, (-1, num, 1))
        U1_vali = np.reshape(u1_vali, (-1, num, 1))
        U1_test = np.reshape(u1_test, (-1, num, 1))
        U1_train=(U1_train-u1_train_mean)/(u1_train_std)
        U1_vali=(U1_vali-u1_train_mean)/(u1_train_std)
        U1_test=(U1_test-u1_train_mean)/(u1_train_std)
        
        u2_train_mean = np.mean(u2_train,0)
        u2_train_std = np.std(u2_train,0)

        u2_train_mean = np.reshape(u2_train_mean, (-1, num, 1))
        u2_train_std = np.reshape(u2_train_std, (-1, num, 1))
        U2_train = np.reshape(u2_train, (-1, num, 1))
        U2_vali = np.reshape(u2_vali, (-1, num, 1))
        U2_test = np.reshape(u2_test, (-1, num, 1))
        U2_train=(U2_train-u2_train_mean)/u2_train_std 
        U2_vali=(U2_vali-u2_train_mean)/u2_train_std
        U2_test=(U2_test-u2_train_mean)/u2_train_std
 
        
        return F_train, U1_train, F_vali, U1_vali, F_test, U1_test, x1_train, x1_vali, x1_test, \
            U2_train,  U2_vali,  U2_test, u1_train_mean, u1_train_std, u2_train_mean, u2_train_std
 
                
    def minibatch(self):

        batch_id = np.random.choice(self.F_train.shape[0], self.bs, replace=False)
        time_id = np.arange(self.bs_time)
        
        f_train = [self.F_train[i:i+1] for i in batch_id]
        f_train = np.concatenate(f_train, axis=0)
        
        u1_train = [self.U1_train[i:i+1] for i in batch_id]
        u1_train = np.concatenate(u1_train, axis=0)[:,time_id,:]
        u2_train = [self.U2_train[i:i+1] for i in batch_id]
        u2_train = np.concatenate(u2_train, axis=0)[:,time_id,:]
        
        x1_train = self.X1_train
        
        X1min = np.amin(x1_train,axis=0)
        X1max = np.amax(x1_train,axis=0)

        return x1_train,  f_train,  u1_train, u2_train, X1min, X1max
    
    def alldata(self):

        f_train = self.F_train
        u1_train = self.U1_train
        u2_train = self.U2_train
        
        x1_train = self.X1_train

        
        X1min = np.amin(x1_train,axis=0)
        X1max = np.amax(x1_train,axis=0)

 

        return x1_train, f_train,  u1_train, u2_train, X1min, X1max  

    def valibatch(self, num_vali):

        batch_id = np.arange(num_vali)
        
        f_vali = [self.F_vali[i:i+1] for i in batch_id]
        f_vali = np.concatenate(f_vali, axis=0)
        
        u1_vali = [self.U1_vali[i:i+1] for i in batch_id]
        u1_vali = np.concatenate(u1_vali, axis=0)

        u2_vali = [self.U2_vali[i:i+1] for i in batch_id]
        u2_vali = np.concatenate(u2_vali, axis=0)
        
        x1_vali = self.X1_vali     
        
        return x1_vali, f_vali, u1_vali, u2_vali

    def testbatch(self, num_test):

        batch_id = np.arange(num_test)
        
        f_test = [self.F_test[i:i+1] for i in batch_id]
        f_test = np.concatenate(f_test, axis=0)
        
        u1_test = [self.U1_test[i:i+1] for i in batch_id]
        u1_test = np.concatenate(u1_test, axis=0)
        u2_test = [self.U2_test[i:i+1] for i in batch_id]
        u2_test = np.concatenate(u2_test, axis=0)
                
        x1_test = self.X1_test
   
        
        return x1_test, f_test, u1_test, u2_test













