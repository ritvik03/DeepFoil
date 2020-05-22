import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
# from google.colab import files
# files.upload()
from plotting import newfig, savefig
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

class DeepVIV:
    # Initialize the class
    def __init__(self, t, x, y,
                       u, v, eta,
                       layers_uvp, layers_eta,
                       Re):

        self.Re = Re

        X = np.concatenate([t, x, y], 1)
        self.X_min = X.min(0)
        self.X_max = X.max(0)

        # data on velocity (inside the domain)
        self.t = t
        self.x = x
        self.y = y
        self.u = u
        self.v = v
        self.eta = eta

        # layers
        self.layers_uvp = layers_uvp
        self.layers_eta  = layers_eta

        # initialize NN
        self.weights_uvp, self.biases_uvp = self.initialize_NN(layers_uvp)
        self.weights_eta, self.biases_eta = self.initialize_NN(layers_eta)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))

        # placeholders for data on velocities (inside the domain)
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.eta_tf = tf.placeholder(tf.float32, shape=[None, 1])

        self.dummy_tf = tf.placeholder(tf.float32, shape=(None, layers_uvp[-1])) # dummy variable for fwd_gradients

        # physics informed neural networks (inside the domain)
        (self.u_pred,
         self.v_pred,
         self.p_pred,
         self.eta_pred,
         self.eq1_pred,
         self.eq2_pred,
         self.eq3_pred) = self.net_VIV(self.t_tf, self.x_tf, self.y_tf)

        # loss
        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.v_tf - self.v_pred)) + \
                    tf.reduce_mean(tf.square(self.eta_tf - self.eta_pred)) + \
                    tf.reduce_mean(tf.square(self.eq1_pred)) + \
                    tf.reduce_mean(tf.square(self.eq2_pred)) + \
                    tf.reduce_mean(tf.square(self.eq3_pred))

        # optimizers
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = X
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def fwd_gradients(self, U, x):
        g = tf.gradients(U, x, grad_ys=self.dummy_tf)[0]
        return tf.gradients(g, self.dummy_tf)[0]

    def net_VIV(self, t, x, y):
        X = 2.0*(tf.concat([t,x,y], 1) - self.X_min)/(self.X_max - self.X_min) - 1.0
        uvp = self.neural_net(X, self.weights_uvp, self.biases_uvp)

        t_tmp = 2.0*(t - self.X_min[0])/(self.X_max[0] - self.X_min[0]) - 1
        eta = self.neural_net(t_tmp, self.weights_eta, self.biases_eta)

        uvp_t = self.fwd_gradients(uvp, t)
        uvp_x = self.fwd_gradients(uvp, x)
        uvp_y = self.fwd_gradients(uvp, y)
        uvp_xx = self.fwd_gradients(uvp_x, x)
        uvp_yy = self.fwd_gradients(uvp_y, y)

        eta_t = tf.gradients(eta, t)[0]
        eta_tt = tf.gradients(eta_t, t)[0]

        u = uvp[:,0:1]
        v = uvp[:,1:2]
        p = uvp[:,2:3]

        u_t = uvp_t[:,0:1]
        v_t = uvp_t[:,1:2]

        u_x = uvp_x[:,0:1]
        v_x = uvp_x[:,1:2]
        p_x = uvp_x[:,2:3]

        u_y = uvp_y[:,0:1]
        v_y = uvp_y[:,1:2]
        p_y = uvp_y[:,2:3]

        u_xx = uvp_xx[:,0:1]
        v_xx = uvp_xx[:,1:2]

        u_yy = uvp_yy[:,0:1]
        v_yy = uvp_yy[:,1:2]

        eq1 = u_t + (u*u_x + v*u_y) + p_x - (1.0/self.Re)*(u_xx + u_yy)
        eq2 = v_t + (u*v_x + v*v_y) + p_y - (1.0/self.Re)*(v_xx + v_yy) + eta_tt
        eq3 = u_x + v_y

        return u, v, p, eta, eq1, eq2, eq3

    def train(self, num_epochs, batch_size, learning_rate):

        path = ''

        saver = tf.train.Saver()

        try:
            saver = tf.train.import_meta_graph(path+'trained_models/model.ckpt.meta')
            saver.restore(self.sess,tf.train.latest_checkpoint(path+'trained_models'))
        except:
            print("[INFO] Cannot import the pre-trained model. Now training on fresh model.")

        for epoch in range(num_epochs):

            N = self.t.shape[0]
            perm = np.random.permutation(N)

            start_time = time.time()
            for it in range(0, N, batch_size):
                idx = perm[np.arange(it,it+batch_size)]
                (t_batch,
                 x_batch,
                 y_batch,
                 u_batch,
                 v_batch,
                 eta_batch) = (self.t[idx,:],
                               self.x[idx,:],
                               self.y[idx,:],
                               self.u[idx,:],
                               self.v[idx,:],
                               self.eta[idx,:])

                tf_dict = {self.t_tf: t_batch, self.x_tf: x_batch, self.y_tf: y_batch,
                           self.u_tf: u_batch, self.v_tf: v_batch, self.eta_tf: eta_batch,
                           self.dummy_tf: np.ones((batch_size, self.layers_uvp[-1])),
                           self.learning_rate: learning_rate}

                self.sess.run(self.train_op, tf_dict)

                # Print
                if it % (10*batch_size) == 0:
                    elapsed = time.time() - start_time
                    loss_value, learning_rate_value = self.sess.run([self.loss,self.learning_rate], tf_dict)
                    print('Epoch: %d, It: %d, Loss: %.3e, Time: %.2f, Learning Rate: %.3e'
                          %(epoch, it/batch_size, loss_value, elapsed, learning_rate_value))
                    start_time = time.time()
            save_path = saver.save(self.sess, path+'Results/trained_models/model.ckpt')
            print('Checkpoint saved')


    def predict(self, t_star, x_star, y_star):

        tf_dict = {self.t_tf: t_star, self.x_tf: x_star, self.y_tf: y_star}

        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)
        eta_star = self.sess.run(self.eta_pred, tf_dict)

        return u_star, v_star, p_star, eta_star

    def predict_drag_lift(self, t_cyl):

        viscosity = (1.0/self.Re)

        theta = np.linspace(0.0,2*np.pi,200)[:,None] # N x 1
        d_theta = theta[1,0] - theta[0,0]
        x_cyl = 0.5*np.cos(theta) # N x 1
        y_cyl = 0.5*np.sin(theta) # N x 1

        N = x_cyl.shape[0]
        T = t_cyl.shape[0]
        from google.colab import files
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

def plot_solution(x_star, y_star, u_star, ax):

    nn = 200
    x = np.linspace(x_star.min(), x_star.max(), nn)
    y = np.linspace(y_star.min(), y_star.max(), nn)
    X, Y = np.meshgrid(x,y)

    X_star = np.concatenate((x_star, y_star), axis=1)

    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='linear')

    # h = ax.pcolor(X,Y,U_star, cmap = 'jet')

    h = ax.imshow(U_star, interpolation='nearest', cmap='jet',
                  extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
                  origin='lower', aspect='auto')

    return h
