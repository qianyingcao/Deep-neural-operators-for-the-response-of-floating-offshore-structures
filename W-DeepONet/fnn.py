import tensorflow as tf
import numpy as np

class FNN:
    def __init__(self):
        pass
    
    def hyper_initial(self, layers):
        L = len(layers)
        W = []
        b = []
        for l in range(1, L):
            in_dim = layers[l-1]
            out_dim = layers[l]
            std = np.sqrt(2./(in_dim + out_dim))
            weight = tf.Variable(tf.random_normal(shape=[in_dim, out_dim], stddev=std))
            bias = tf.Variable(tf.zeros(shape=[1, out_dim]))
            W.append(weight)
            b.append(bias)

        return W, b

    def fnn_T1(self, W, b, X, Xmin, Xmax):
        #A =(X - Xmin)/(Xmax - Xmin)
        A = X
        L = len(W)
        for i in range(L-1):
            # A = tf.nn.leaky_relu(tf.add(tf.matmul(A, W[i]), b[i]))
            A = tf.sin(tf.add(tf.matmul(A, W[i]), b[i]))
        Y = tf.add(tf.matmul(A, W[-1]), b[-1])
        
        return Y
    
    def fnn_B1(self, W, b, X):
        A = X
        L = len(W)
        for i in range(L-1):
            # A = tf.nn.leaky_relu(tf.add(tf.matmul(A, W[i]), b[i]))
            A = tf.sin(tf.add(tf.matmul(A, W[i]), b[i]))
        Y = tf.add(tf.matmul(A, W[-1]), b[-1])  
        return Y

    def fnn_T2(self, W, b, X, Xmin, Xmax):
        #A =(X - Xmin)/(Xmax - Xmin)
        A = X
        L = len(W)
        for i in range(L-1):
            A = tf.nn.leaky_relu(tf.add(tf.matmul(A, W[i]), b[i]))
            # A = tf.sin(tf.add(tf.matmul(A, W[i]), b[i]))
        Y = tf.add(tf.matmul(A, W[-1]), b[-1])
        
        return Y
    
    def fnn_B2(self, W, b, X):
        A = X
        L = len(W)
        for i in range(L-1):
            A = tf.nn.leaky_relu(tf.add(tf.matmul(A, W[i]), b[i]))
            # A = tf.sin(tf.add(tf.matmul(A, W[i]), b[i]))
        Y = tf.add(tf.matmul(A, W[-1]), b[-1])  
        return Y

    def fnn_net(self,u_B,u_T,p1,num):
        u1_pred = tf.einsum('ijk, lk->il', u_B[:,:,0:p1//2], u_T[:,0:p1//2])
        u2_pred = tf.einsum('ijk, lk->il', u_B[:,:,p1//2:p1], u_T[:,p1//2:p1])
        u1_pred = tf.reshape(u1_pred,(-1,num,1))
        u2_pred = tf.reshape(u2_pred,(-1,num,1))
        # u_pred = tf.einsum('ijk, lk->ilk', u_B, u_T)
        #u_pred1 = tf.reshape(u_pred0,(-1,1))
        # biases_last = tf.Variable(tf.zeros(1))
        # u_pred1+= biases_last
        #u_pred = tf.reshape(u_pred1,tf.shape(u_pred0))
        return u1_pred, u2_pred
    
    def l2_regularizer(self, W):
        
        l2 = 0.0
        L = len(W)
        for i in range(L-1):
            l2 += tf.nn.l2_loss(W[i])
        
        return l2 