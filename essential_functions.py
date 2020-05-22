#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import keras
from keras import *
from keras.layers import *
from keras.models import *
import keras.backend as K
import numpy as np
import time
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

from scipy.interpolate import griddata
import time
# from google.colab import files
# files.upload()
from plotting import newfig, savefig
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

import math
import matplotlib.pyplot as plt
import numpy as np

from IPython.display import Image


# In[2]:


def custom_loss_wrapper(Re):
#     input_tensor = concatenate([t, x, y], 1)
    def gradient_calc(Re):
        
        uvp = model.output
        u = uvp[:,0:1]
        v = uvp[:,1:2]
        p = uvp[:,2:3]
        eta = uvp[:,3:4]
#         print(u)
        
        u_t,u_x,u_y = K.gradients(u,model.input)
        v_t,v_x,v_y = K.gradients(v,model.input)
        p_t,p_x,p_y = K.gradients(p,model.input)
        eta_t,eta_x,eta_y = K.gradients(eta,model.input)
        
        u_xx = K.gradients(u_x,model.input[1])[0]
        u_yy = K.gradients(u_y,model.input[2])[0]
        v_xx = K.gradients(v_x,model.input[1])[0]
        v_yy = K.gradients(v_y,model.input[2])[0]
        eta_tt = K.gradients(eta_t,model.input[0])[0]
        
#         print((u_xx)+(u_yy))
        
        eq1 = u_t + (u*u_x + v*u_y) + p_x - (1.0/Re)*(u_xx + u_yy)
        eq2 = v_t + (u*v_x + v*v_y) + p_y - (1.0/Re)*(v_xx + v_yy) + eta_tt
        eq3 = u_x + v_y
        
        loss = K.mean(tf.square(eq1)) + K.mean(tf.square(eq2)) + K.mean(tf.square(eq3))
        
#         print((u_xx))
        return loss

    def custom_loss(y_true, y_pred):
        navier_loss = gradient_calc(Re=Re)
#         navier_loss = net_VIV(input_tensor,y_pred,Re=1000)
        return tf.reduce_mean(tf.square(y_true - y_pred)) + navier_loss
    return custom_loss

# model.compile(loss={'out_eta': 'mean_squared_error', 'out_uvp': custom_loss_wrapper(model.layers[3])}, optimizer='adam', metrics=['accuracy','mean_squared_error'])
# model.compile(loss=custom_loss_wrapper(Re=1000), optimizer='adam', metrics=['accuracy','mean_squared_error'])


# In[2]:


def load_data(filename = "final_translated_data.csv"):
    time_1 = time.time()
    data = pd.read_csv(filename)
    print("[INFO] Time taken = "+str(time.time()-time_1)+" seconds.")
    return data


# In[6]:


def scale_learn(t_orig,x_orig,y_orig,u_orig,v_orig,p_orig,eta_orig):
    scale_vals = np.concatenate([t_orig,x_orig,y_orig,u_orig,v_orig,p_orig,eta_orig],1)
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(scale_vals)
    return scaler


# In[4]:


def create_tensors(data):
    
    t_star = data['t'] # T x 1
    eta_star = data['eta'] # T x 1
    
    T = t_star.shape[0]
        
    X_star = data['x']
    Y_star = data['y']        
    U_star = data['u']
    V_star = data['v']
    P_star = data['p']

    t = t_star.to_numpy()
    eta = eta_star.to_numpy()
    
    # T = t_star.shape[0]
        
    x = X_star.to_numpy()
    y = Y_star.to_numpy()        
    u = U_star.to_numpy()
    v = V_star.to_numpy()
    p = P_star.to_numpy()
    
    ## clipping
    
    t = t.reshape((t.shape[0],1))
    x = x.reshape((x.shape[0],1))
    y = y.reshape((y.shape[0],1))
    u = u.reshape((u.shape[0],1))
    v = v.reshape((v.shape[0],1))
    p = p.reshape((p.shape[0],1))
    eta = eta.reshape((eta.shape[0],1))
    
    return t,x,y,u,v,p,eta


# In[7]:


def scale(scaler,t,x,y,u,v,p,eta):
    scale_vals = np.concatenate([t,x,y,u,v,p,eta],1)
    scale_vals = scaler.transform(scale_vals)
    scaled_list = np.split(scale_vals,7,axis=1)
    
    t_scaled = scaled_list[0]
    x_scaled = scaled_list[1]
    y_scaled = scaled_list[2]
    u_scaled = scaled_list[3]
    v_scaled = scaled_list[4]
    p_scaled = scaled_list[5]
    eta_scaled = scaled_list[6]

    return t_scaled,x_scaled,y_scaled,u_scaled,v_scaled,p_scaled,eta_scaled


# In[8]:


def de_scale(scaler,t_test,x_test,y_test,uvpEta_test):
    
    final_test = np.concatenate([t_test,x_test,y_test,uvpEta_test],1) 
    de_scale_vals = scaler.inverse_transform(final_test)
    scaled_list = np.split(de_scale_vals,7,axis=1)
    
    t_test_invSc = scaled_list[0]
    x_test_invSc = scaled_list[1]
    y_test_invSc = scaled_list[2]
    u_test_invSc = scaled_list[3]
    v_test_invSc = scaled_list[4]
    p_test_invSc = scaled_list[5]
    eta_test_invSc = scaled_list[6]
    
    return t_test_invSc,x_test_invSc,y_test_invSc,u_test_invSc,v_test_invSc,p_test_invSc,eta_test_invSc


# In[3]:


def plot_time_data(time_snap,data):
  time_data = data[data['t']==time_snap]
  x_snap = time_data["x"]
  y_snap = time_data["y"]
  plt.plot(x_snap, y_snap, 'bo',markersize=0.1,marker='o',scalex=3,scaley=3)
  plt.title("at time: "+str(time_snap)+" second")
  plt.savefig("data_"+str(time_snap)+"_sec.png")
  plt.show()


# In[9]:


def plot_solution(x_star, y_star, u_star, ax):

    nn = 200
    x = np.linspace(x_star.min(), x_star.max(), nn)
    y = np.linspace(y_star.min(), y_star.max(), nn)
    X, Y = np.meshgrid(x,y)

    X_star = np.concatenate((x_star, y_star), axis=1)

    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='linear', rescale=True)

    # h = ax.pcolor(X,Y,U_star, cmap = 'jet')

    h = ax.imshow(U_star, interpolation='nearest', cmap='jet', vmin=-2, vmax=2,
                  extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
                  origin='lower', aspect='auto')

    return h


# In[14]:


class airfoil_shape():
    
    def __init__(self):
        self.m = 0.03
        self.p = 0.4
        self.t = 0.12
        self.c = 1.0
    
    def camber_line( self,x ):
        return np.where((x>=0)&(x<=(self.c*self.p)),self.m*(x / np.power(self.p,2)) * (2.0 * self.p - (x / self.c)),self.m * ((self.c - x) / np.power(1-self.p,2)) * (1.0 + (x / self.c) - 2.0 * self.p ))

    def dyc_over_dx( self,x ):
        return np.where((x>=0)&(x<=(self.c*self.p)),((2.0 * self.m) / np.power(self.p,2)) * (self.p - x / self.c),((2.0 * self.m ) / np.power(1-self.p,2)) * (self.p - x / self.c ))

    def thickness( self,x ):
        term1 =  0.2969 * (np.sqrt(x/self.c))
        term2 = -0.1260 * (x/self.c)
        term3 = -0.3516 * np.power(x/self.c,2)
        term4 =  0.2843 * np.power(x/self.c,3)
        term5 = -0.1015 * np.power(x/self.c,4)
        return 5 * self.t * self.c * (term1 + term2 + term3 + term4 + term5)

    def naca4(self,x):
        dyc_dx = self.dyc_over_dx(x)
        th = np.arctan(dyc_dx)
        yt = self.thickness(x)
        yc = self.camber_line(x)  
        return ((x - yt*np.sin(th), yc + yt*np.cos(th)),(x + yt*np.sin(th), yc - yt*np.cos(th)))
    


# In[1]:


def plot_airfoil(ax):
    x = np.linspace(0,1,200)
    foil = airfoil_shape()
    for item in foil.naca4(x):
        x_val = item[0]*0.9953705935+item[1]*0.09611129781
        y_val = 0.05 -item[0]*0.09611129781+item[1]*0.9953705935
        ax.fill(x_val, y_val, 'w')
        ax.plot(x_val,y_val,'k')

def plot_figs(t_test,x_test,y_test,u_test,v_test,p_test,u_pred,v_pred,p_pred):
    
    fig, ax = newfig(2.0, 1.0)
    ax.axis('off')
    fig.suptitle("Time: "+str(t_test[0][0]))

    gs = gridspec.GridSpec(3, 3)
    gs.update(wspace=0.5, hspace=0.5)
        
    ########      Exact u(t,x,y)     ###########     
    ax = plt.subplot(gs[0:1, 0])
    h = plot_solution(x_test,y_test,u_test,ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
#     ax.add_artist(circle21)
    plot_airfoil(ax)

    ax.axis('equal')
    
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Learned $u(t,x,y)$', fontsize = 10)
    
    ########     Learned u(t,x,y)     ###########
    ax = plt.subplot(gs[0:1, 1])
    h = plot_solution(x_test,y_test,u_pred,ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
#     ax.add_artist(circle22)
    plot_airfoil(ax)
    ax.axis('equal')
    
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Exact $u(t,x,y)$', fontsize = 10)
    
    ########     Difference u(t,x,y)     ###########
    ax = plt.subplot(gs[0:1, 2])
    h = plot_solution(x_test,y_test,u_pred-u_test,ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
#     ax.add_artist(circle22)
    plot_airfoil(ax)
    ax.axis('equal')
    
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Error $u(t,x,y)$', fontsize = 10)
    
    ########      Exact v(t,x,y)     ###########     
    ax = plt.subplot(gs[1:2, 0])
    h = plot_solution(x_test,y_test,v_test,ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
#     ax.add_artist(circle31)
    ax.axis('equal')
    plot_airfoil(ax)
    
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Learned $v(t,x,y)$', fontsize = 10)
    
    ########     Learned v(t,x,y)     ###########
    ax = plt.subplot(gs[1:2, 1])
    h = plot_solution(x_test,y_test,v_pred,ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
#     ax.add_artist(circle32)
    ax.axis('equal')
    plot_airfoil(ax)
    
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Exact $v(t,x,y)$', fontsize = 10)
    
    ########     Difference v(t,x,y)     ###########
    ax = plt.subplot(gs[1:2, 2])
    h = plot_solution(x_test,y_test,v_pred-v_test,ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
#     ax.add_artist(circle22)
    plot_airfoil(ax)
    ax.axis('equal')
    
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Error $v(t,x,y)$', fontsize = 10)
    
    ########      Exact p(t,x,y)     ###########     
    ax = plt.subplot(gs[2:3, 0])
    h = plot_solution(x_test,y_test,p_test,ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
#     ax.add_artist(circle41)
    ax.axis('equal')
    plot_airfoil(ax)
    
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Learned $p(t,x,y)$', fontsize = 10)
    
    ########     Learned p(t,x,y)     ###########
    ax = plt.subplot(gs[2:3, 1])
    h = plot_solution(x_test,y_test,p_pred,ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
#     ax.add_artist(circle42)
    ax.axis('equal')
    plot_airfoil(ax)
        
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Exact $p(t,x,y)$', fontsize = 10)
    
    ########     Difference p(t,x,y)     ###########
    ax = plt.subplot(gs[2:3, 2])
    h = plot_solution(x_test,y_test,p_pred-p_test,ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
#     ax.add_artist(circle22)
    plot_airfoil(ax)
    ax.axis('equal')
    
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Error $p(t,x,y)$', fontsize = 10)    
    
    savefig('Figures/VIV_data_on_velocities', crop = False)


# In[16]:


def rmse(y_true,y_pred):
    rmse_error = sqrt(mean_squared_error(y_true,y_pred))
    return rmse_error
    
def time_snap(time,scaler,data,model,plot=False,print_error=False):
    
    data_snap = data[data['t']==time]
    
    t_snap,x_snap,y_snap,u_snap,v_snap,p_snap,eta_snap = create_tensors(data_snap)
    t_test,x_test,y_test,u_test,v_test,p_test,eta_test = scale(scaler,t_snap,x_snap,y_snap,u_snap,v_snap,p_snap,eta_snap)
    uvpEta_test = model.predict([t_test,x_test,y_test])    
    t_pred,x_pred,y_pred,u_pred,v_pred,p_pred,eta_pred = de_scale(scaler,t_test,x_test,y_test,uvpEta_test)
    
    rmse_u = rmse(u_snap,u_pred)
    rmse_v = rmse(v_snap,v_pred)
    rmse_p = rmse(p_snap,p_pred)
    
    
    if (print_error):
        print("RMSE for u: "+str((rmse_u)*100/scaler.data_range_[3])+" %")
        print("RMSE for v: "+str((rmse_v)*100/scaler.data_range_[4])+" %")
        print("RMSE for p: "+str((rmse_p)*100/scaler.data_range_[5])+" %")
    
    if (plot):
        plot_figs(t_pred,x_pred,y_pred,u_pred,v_pred,p_pred,u_snap,v_snap,p_snap)
#         u_pred_i,v_pred_i,p_pred_i,eta_pred_i = np.split(uvpEta_test,4,axis=1)
#         plot_figs(t_test,x_test,y_test,u_test,v_test,p_test,u_pred_i,v_pred_i,p_pred_i)
    
    return u_pred,v_pred,p_pred,eta_pred[0][0]


# In[17]:


def plot3D(x,y,z,s,saved=False):
    if(saved):
        image = Image(filename='plot3d_show.png')
        display(image)
        return 
    fig = plt.figure(figsize=(25,25))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='k', marker='.',s=s)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('t')
    fig.savefig("plot3d.png")
    fig.show()
    return fig
#     plt.show()


# In[2]:


def get_data(data,timeVal):
    data_sub = data[data['t']==timeVal]
    return np.array(data_sub['u']).reshape((len(data_sub['u']),1)),np.array(data_sub['v']).reshape((len(data_sub['u']),1)),np.array(data_sub['p']).reshape((len(data_sub['u']),1)),np.array(data_sub['eta']).reshape((len(data_sub['u']),1)),np.array(data_sub['x']).reshape((len(data_sub['u']),1)),np.array(data_sub['y']).reshape((len(data_sub['u']),1))

def get_time_plot(timeVal,data,scaler,model):
    u_pred,v_pred,p_pred,eta_pred=time_snap(timeVal,scaler,data,model,plot=False,print_error=False)
    u_test,v_test,p_test,eta_test,x_test,y_test = get_data(data,timeVal)
    t_test = np.full((u_test.shape[0], 1), 7)
    fig = plot_figs(t_test,x_test,y_test,u_test,v_test,p_test,u_pred,v_pred,p_pred)
    return fig,rmse(u_pred,u_test),rmse(v_pred,v_test),rmse(p_pred,p_test)


# In[ ]:


def predict_drag_lift(self, t_cyl):
    
    viscosity = (1.0/self.Re)
    
    theta = np.linspace(0.0,2*np.pi,200)[:,None] # N x 1
    d_theta = theta[1,0] - theta[0,0]
    x_cyl = 0.5*np.cos(theta) # N x 1
    y_cyl = 0.5*np.sin(theta) # N x 1
    
    N = x_cyl.shape[0]
    T = t_cyl.shape[0]
    
    T_star = np.tile(t_cyl, (1,N)).T # N x T
    X_star = np.tile(x_cyl, (1,T)) # N x T
    Y_star = np.tile(y_cyl, (1,T)) # N x T
    
    t_star = np.reshape(T_star,[-1,1]) # NT x 1
    x_star = np.reshape(X_star,[-1,1]) # NT x 1
    y_star = np.reshape(Y_star,[-1,1]) # NT x 1

    u_x_pred = tf.gradients(self.u_pred, self.x_tf)[0]
    u_y_pred = tf.gradients(self.u_pred, self.y_tf)[0]
    
    v_x_pred = tf.gradients(self.v_pred, self.x_tf)[0]
    v_y_pred = tf.gradients(self.v_pred, self.y_tf)[0]
    
    tf_dict = {self.t_tf: t_star, self.x_tf: x_star, self.y_tf: y_star}
    
    p_star, u_x_star, u_y_star, v_x_star, v_y_star = self.sess.run([self.p_pred, u_x_pred, u_y_pred, v_x_pred, v_y_pred], tf_dict)
    
    P_star = np.reshape(p_star, [N,T]) # N x T
    P_star = P_star - np.mean(P_star, axis=0)
    U_x_star = np.reshape(u_x_star, [N,T]) # N x T
    U_y_star = np.reshape(u_y_star, [N,T]) # N x T
    V_x_star = np.reshape(v_x_star, [N,T]) # N x T
    V_y_star = np.reshape(v_y_star, [N,T]) # N x T

    INT0 = (-P_star[0:-1,:] + 2*viscosity*U_x_star[0:-1,:])*X_star[0:-1,:] + viscosity*(U_y_star[0:-1,:] + V_x_star[0:-1,:])*Y_star[0:-1,:]
    INT1 = (-P_star[1: , :] + 2*viscosity*U_x_star[1: , :])*X_star[1: , :] + viscosity*(U_y_star[1: , :] + V_x_star[1: , :])*Y_star[1: , :]
        
    F_D = 0.5*np.sum(INT0.T+INT1.T, axis = 1)*d_theta # T x 1

    
    INT0 = (-P_star[0:-1,:] + 2*viscosity*V_y_star[0:-1,:])*Y_star[0:-1,:] + viscosity*(U_y_star[0:-1,:] + V_x_star[0:-1,:])*X_star[0:-1,:]
    INT1 = (-P_star[1: , :] + 2*viscosity*V_y_star[1: , :])*Y_star[1: , :] + viscosity*(U_y_star[1: , :] + V_x_star[1: , :])*X_star[1: , :]
        
    F_L = 0.5*np.sum(INT0.T+INT1.T, axis = 1)*d_theta # T x 1
        
    return F_D, F_L

