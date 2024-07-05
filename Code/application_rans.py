import os
import tensorflow as tf
import keras 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import scipy.optimize
import pandas as pd 
from keras import optimizers
import tensorflow_probability as tfp
import math 
from mat4py import loadmat
import random 
import matplotlib.ticker as ticker
import pickle
import cmasher as cmr
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats as st
from matplotlib.ticker import FuncFormatter


tfpl = tfp.layers
tfd = tfp.distributions


with open('ubar.pkl', 'rb') as f:
    ubar = pickle.load(f)
with open('vbar.pkl', 'rb') as f:
    vbar = pickle.load(f)
with open('uvbar.pkl', 'rb') as f:
    uvbar = pickle.load(f)
with open('u1000.pkl', 'rb') as f:
    u1000 = pickle.load(f)


xx = np.loadtxt('xgrid.txt')
yy = np.loadtxt('ygrid.txt')
n = xx.shape[0]*xx.shape[1]
xspace = xx[0,:]
yspace = yy[:,0]

data = pd.DataFrame(xx.reshape(n, order='F'), columns=['x'])
data['y'] = yy.reshape(n, order='F')
data['u'] = ubar.reshape(n, order='F')
data['uv'] = uvbar.reshape(n, order='F')

xy = data[['x','y']].to_numpy()


# FIGURE 3; u, ubar, ubar(y)
fig, ax = plt.subplots(3,1,figsize=(20,18), gridspec_kw={'height_ratios': [1.7, 1.4, 1.4]})
plt.subplots_adjust(wspace=0.05, hspace=0.25)

vmin = np.min([ubar.min(), u[:,:,999].min()])
vmax = np.max([ubar.max(), u[:,:,999].max()])
cmap = cmr.get_sub_cmap('Reds', 0, .8)

np.random.seed(1)
n_y = int(0.05*len(yspace))
x_dots = np.random.choice(xspace, 5, replace=False)
y_dots = []

for i in range(5):
    y_dots_temp = np.random.choice(len(yspace), n_y, replace=False)
    y_dots.append(y_dots_temp)

im = ax[0].imshow(u1000, extent=[xspace[0], xspace[-1], yspace[0], yspace[-1]],\
                cmap=cmap, interpolation='nearest', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
#ax[0].set_title(r'$u(x,y,1000)$', fontsize=20)
ax[0].set_xlabel(r'$x$ (length of the PIV measurement window, in $m$)', fontsize=16)
ax[0].set_ylabel(r'$y$ (distance from wall, in $m$)', fontsize=16)
ax[0].tick_params(axis='y', labelsize=18)
ax[0].tick_params(axis='x', labelsize=18)
ax[0].annotate("(a)", xy=(-0.06, 1.03), xycoords="axes fraction", fontsize=18)

im = ax[1].imshow(ubar, extent=[xspace[0], xspace[-1], yspace[0], yspace[-1]],\
                cmap=cmap, interpolation='nearest', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
#ax[1].set_title(r'$\overline{u}(x,y)$', fontsize=20)
ax[1].set_ylabel(r'$y$ (distance from wall, in $m$)', fontsize=16)
ax[1].set_xlabel(r'$x$ (length of the PIV measurement window, in $m$)', fontsize=16)
ax[1].tick_params(axis='y', labelsize=18)
ax[1].tick_params(axis='x', labelsize=18)
ax[1].annotate("(b)", xy=(-0.06, 1.03), xycoords="axes fraction", fontsize=18)

for i in range(5):
    ax[1].scatter(np.tile(x_dots[i], reps=len(y_dots[i])), yspace[y_dots[i]], marker='x', color='black')

norm = Normalize(vmin=vmin, vmax=vmax)
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("top", size="10%", pad='10%', aspect=0.02)
cb = plt.colorbar(im, cax=cax, orientation='horizontal', shrink=10, aspect=15) #, shrink=0.6, pad=0.15)
cb.set_ticks(np.round(np.linspace(vmin, vmax, 7),2))
cb.set_label(label=r'$m/s$', size=20)
cb.ax.tick_params(labelsize=20)

cax.xaxis.set_ticks_position("top")
cax.xaxis.set_label_position("top")

ax[2].plot(ubar.mean(axis=1), yspace, 'red')
ax[2].set_ylabel(r'$y$ (distance from wall, in $m$)', fontsize=16)
ax[2].set_xlabel(r'$\overline{u}(y)$', fontsize=16)
ax[2].tick_params(axis='y', labelsize=18)
ax[2].tick_params(axis='x', labelsize=18)
ax[2].annotate("(c)", xy=(-0.06, 1.03), xycoords="axes fraction", fontsize=18);

#plt.tight_layout();


# FIGURE S10; SHOWING VBAR / UBAR
from matplotlib import cm
from matplotlib.colors import ListedColormap

fig, ax = plt.subplots(1,1,figsize=(20,9), sharex=True)

vmin = np.abs(vbar/ubar).min()
vmax = np.abs(vbar/ubar).max()

top = cm.get_cmap('Reds', 128)
bottom = cm.get_cmap('Reds', 128)
newcolors = np.vstack((top(np.linspace(1, 0, 128)),
                       bottom(np.linspace(0, 1, 128))))
rwr = ListedColormap(newcolors, name='RedWhiteRed')

fmt = lambda x, pos: '{:.4f}'.format(x)
im1 = ax.contourf(xx, yy, np.abs(vbar/ubar), levels=100, cmap='Greys', vmin=vmin, vmax=vmax)
ax.set_title(r'$\left|\frac{\overline{v}(x,y)}{\overline{u}(x,y)}\right|$', fontsize=20)
ax.set_ylabel(r'$y$ (distance from wall, in $m$)', fontsize=18)
ax.set_xlabel(r'$x$ (length of the PIV measurement window, in $m$)', fontsize=18)
ax.tick_params(axis='y', labelsize=18)
ax.tick_params(axis='x', labelsize=18)

for i in range(5):
    ax.scatter(np.tile(x_dots[i], reps=len(y_dots[i])), yspace[y_dots[i]], marker='x', color='black')

cb = fig.colorbar(im1, ax=ax, orientation='horizontal', pad=0.17, aspect=15, shrink=0.6, format=FuncFormatter(fmt))
cb.set_ticks(np.linspace(vmin, vmax, 7))
cb.set_label(label=r'$m/s$', size=18)
cb.ax.tick_params(labelsize=18)

plt.show();

# FIGURE S9; SHOWING GRADIENTS
fig, ax = plt.subplots(2,1,figsize=(20,12), gridspec_kw={'height_ratios': [1.7, 1.46]})
plt.subplots_adjust(wspace=0.05, hspace=0.3)

dudx = np.gradient(ubar, axis=1) / np.gradient(xx, axis=1)
dvdx = np.gradient(vbar, axis=1) / np.gradient(xx, axis=1)

cmap = cmr.get_sub_cmap('coolwarm', 0, .605)
vmin = np.min([dudx.min(),dvdx.min()])
vmax = np.max([dudx.max(),dvdx.max()])

im1 = ax[0].imshow(dudx, extent=[xspace[0], xspace[-1], yspace[0], yspace[-1]],\
                cmap=cmap, interpolation='nearest', origin='lower', aspect='equal', vmin=vmin, vmax=vmax)
ax[0].set_title(r'$\partial \overline{u} / \partial x$', fontsize=20)
ax[0].set_ylabel(r'$y$ (distance from wall, in $m$)', fontsize=18)
#ax[0].set_xlabel(r'$x$ (length of the PIV measurement window, in $m$)', fontsize=18)
ax[0].tick_params(axis='y', labelsize=18)
ax[0].tick_params(axis='x', labelsize=18)
ax[0].annotate("(a)", xy=(-0.06, 1.03), xycoords="axes fraction", fontsize=18);


im2 = ax[1].imshow(dvdx, extent=[xspace[0], xspace[-1], yspace[0], yspace[-1]],\
                cmap=cmap, interpolation='nearest', origin='lower', aspect='equal', vmin=vmin, vmax=vmax)
ax[1].set_title(r'$\partial \overline{v} / \partial x$', fontsize=20)
ax[1].set_ylabel(r'$y$ (distance from wall, in $m$)', fontsize=18)
ax[1].set_xlabel(r'$x$ (length of the PIV measurement window, in $m$)', fontsize=18)
ax[1].tick_params(axis='y', labelsize=18)
ax[1].tick_params(axis='x', labelsize=18)
ax[1].annotate("(b)", xy=(-0.06, 1.03), xycoords="axes fraction", fontsize=18);

norm = Normalize(vmin=vmin, vmax=vmax)
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("top", size="4%", pad='12%', aspect=0.5)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', pad=0.15, aspect=10)
cb.set_ticks(np.round(np.linspace(vmin, vmax, 7), 2))
cb.set_label(label=r'$m/s$', size=20)
cb.ax.tick_params(labelsize=20)
cax.xaxis.set_ticks_position("top")
cax.xaxis.set_label_position("top")

plt.show();


class Network:

    def build(self, layers, num_inputs, num_outputs, activation):

        inputs = tf.keras.layers.Input(shape=(num_inputs,))
        x = inputs
        for layer in layers:
            x = tf.keras.layers.Dense(layer, activation=activation, kernel_initializer='he_normal')(x)

        outputs = tf.keras.layers.Dense(num_outputs, kernel_initializer='he_normal')(x)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs)


# In[9]:


class GradientLayer(tf.keras.layers.Layer):

    def __init__(self, model, **kwargs):
        self.model = model
        super().__init__(**kwargs)

    def call(self, xy):

        x, y = [ xy[..., i, tf.newaxis] for i in range(xy.shape[-1]) ]
        with tf.GradientTape(persistent=True) as gg:
            gg.watch(xy)
            with tf.GradientTape(persistent=True) as g:
                g.watch(xy)
                outputs = self.model(xy)
                u = outputs[..., 0, tf.newaxis]
                uv = outputs[..., 1, tf.newaxis]
            u_y = g.batch_jacobian(u, xy)[..., 1]
            u_x = g.batch_jacobian(u, xy)[..., 0]
            uv_y = g.batch_jacobian(uv, xy)[..., 1]
            del g
        u_yy = gg.batch_jacobian(u_y, xy)[..., 1]
        del gg

        u_grads = u, u_x, u_y, u_yy, uv, uv_y

        return u_grads, y


# In[10]:


class PINN:

    def __init__(self, network, nu, kappa, u_tau, delta):

        self.network = network
        self.nu = nu
        self.kappa = kappa
        self.u_tau = u_tau
        self.delta = delta
        self.grads = GradientLayer(self.network)

    def build(self):

        xy_eqn = tf.keras.layers.Input(shape=(2,))
        xy_uv = tf.keras.layers.Input(shape=(2,))
        xy_bnd = tf.keras.layers.Input(shape=(2,))

        # NS BL equation
        u_grads, y = self.grads(xy_eqn)
        u, u_x, u_y, u_yy, uv, uv_y = u_grads
        
        u_eqn = (self.u_tau**2)/self.delta + (self.nu)*(u_yy) - uv_y
        uv_eqn = uv + self.kappa * (u / self.u_tau) * y * u_y

        # boundary conditions
        u_grads_bnd, _ = self.grads(xy_bnd)

        return tf.keras.models.Model( inputs=[xy_eqn, xy_bnd], outputs=[tf.concat([u_eqn, u_grads_bnd[0]], axis=-1), tf.concat([uv_eqn, uv_eqn], axis=-1)] )

# In[11]:


class L_BFGS_B:
   
    def __init__(self, model, x_train, y_train, maxiter, factr=10, pgtol=1e-10, m=50, maxls=100):

        # set attributes
        self.model = model
        self.x_train = [ tf.constant(x, dtype=tf.float32) for x in x_train ]
        self.y_train = [ tf.constant(y, dtype=tf.float32) for y in y_train ]
        self.factr = factr
        self.pgtol = pgtol
        self.m = m
        self.maxls = maxls
        self.maxiter = maxiter
        self.metrics = ['loss']
        # initialize the progress bar
        self.progbar = tf.keras.callbacks.ProgbarLogger(count_mode='steps', stateful_metrics=self.metrics)
        self.progbar.set_params({'verbose':1, 'epochs':1, 'steps':self.maxiter, 'metrics':self.metrics})

    def set_weights(self, flat_weights):
        shapes = [ w.shape for w in self.model.get_weights() ]
        # compute splitting indices
        split_ids = np.cumsum([ np.prod(shape) for shape in [0] + shapes ])
        # reshape weights
        weights = [ flat_weights[from_id:to_id].reshape(shape)
            for from_id, to_id, shape in zip(split_ids[:-1], split_ids[1:], shapes) ]
        self.model.set_weights(weights)

    @tf.function
    def tf_evaluate(self, x, y):
        with tf.GradientTape() as g:
            loss = tf.reduce_mean(tf.keras.losses.mse(self.model(x), y))
        grads = g.gradient(loss, self.model.trainable_variables)
        return loss, grads

    def evaluate(self, weights):
        self.set_weights(weights)
        loss, grads = self.tf_evaluate(self.x_train, self.y_train)
        loss = loss.numpy().astype('float64')
        grads = np.concatenate([ g.numpy().flatten() for g in grads ]).astype('float64')
        return loss, grads

    def callback(self, weights):
        self.progbar.on_batch_begin(1)
        loss, _ = self.evaluate(weights)
        self.progbar.on_batch_end(1, logs=dict(zip(self.metrics, [loss])))

    def fit(self):
        initial_weights = np.concatenate([ w.flatten() for w in self.model.get_weights() ])
        print('Optimizer: L-BFGS-B (maxiter={})'.format(self.maxiter))
        self.progbar.on_train_begin()
        self.progbar.on_epoch_begin(1)
        scipy.optimize.fmin_l_bfgs_b(func=self.evaluate, x0=initial_weights,
            factr=self.factr, pgtol=self.pgtol, m=self.m,
            maxls=self.maxls, maxiter=self.maxiter, callback=self.callback, iprint=0)
        self.progbar.on_epoch_end(1)
        self.progbar.on_train_end()


# In[12]:


def generate_io(x0, x1, y0, y1, u0, delta, u_tau, nx, ny):

    # INPUTS
    xyt_eqn = np.array(np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))).T.reshape(-1,2)

    xyt_bnd = np.array(np.meshgrid(np.linspace(x0, x1, nx), np.array([y0, y1]))).T.reshape(-1,2)
    xyt_bnd = np.tile(xyt_bnd, (xyt_eqn.shape[0]//xyt_bnd.shape[0], 1))

    x_train = [xyt_eqn, xyt_bnd]

    # OUTPUTS
    uv_ns = np.zeros((nx*ny, 2))
    uv_ns[..., 1] = np.where(xyt_bnd[...,1] == y0, 0, u0)
    zeros = np.zeros((nx*ny, 2))

    y_train = [uv_ns, zeros]

    return x_train, y_train


# In[13]:


def train_ns_priors(delta, u_tau, nu, kappa, layers, x0, x1, y0, y1, u0, nx, ny, maxiter, factr):

    print('Building PINN...')
    keras.utils.set_random_seed(812)
    xy_train, u_train = generate_io(x0, x1, y0, y1, u0, delta=delta, u_tau=u_tau, nx=nx, ny=ny)
    network = Network().build(layers=layers, num_inputs=xy_train[0].shape[1], num_outputs=u_train[0].shape[1], activation='tanh')
    pinn = PINN(network=network, nu=nu, kappa=kappa, u_tau=u_tau, delta=delta).build()
    print('PINN built.')
    print()

    print('Beginning training...')
    lbfgs = L_BFGS_B(pinn, xy_train, u_train, maxiter=maxiter, factr=factr)
    lbfgs.fit()
    print('Training complete.')
    print()

    return network, pinn, xy_train, u_train

def get_prior_means(net):
    weights = []
    for l in net.layers[1:]:
        temp = np.append(l.get_weights()[0].flatten(), l.get_weights()[1].flatten())
        weights.append(temp)
    return weights


# In[14]:


delta = 0.1
u_tau = 0.027
u0 = 0.67
Re = 2700
nu = u_tau * delta / Re
kappa = 0.01107

x0 = xspace[0]
x1 = xspace[-1]
y0 = 0
y1 = 0.1 
y_prior = np.linspace(y0, y1, len(yspace))

nx = ubar.shape[1]
ny = ubar.shape[0]


# In[15]:


layers = [10,10]
maxiter = 10000
factr = 1

nn, pinn, xy_train, u_train = train_ns_priors(delta=delta, u_tau=u_tau, nu=nu, kappa=kappa, layers=layers,\
                                            x0=x0, x1=x1, y0=y0, y1=y1, u0=u0, nx=nx, ny=ny, maxiter=maxiter, factr=factr)


# In[16]:


prior_means = get_prior_means(net=nn)

def get_prior_wind(net, x0, x1, y0, y1, ubar=ubar, yspace=yspace, nx=nx, ny=ny, plot=True):
    # create meshgrid coordinates (x, y) for test plots
    x = np.linspace(x0, x1, nx)
    y = np.linspace(y0, y1, ny)
    xyt = np.array(np.meshgrid(x, y)).T.reshape(-1,2)

    # get predictions
    u_pred = net.predict(xyt, verbose=0)
    u = u_pred[:,0].reshape(ubar.shape, order='F')

    fig, ax2 = plt.subplots(1,1,figsize=(15, 8))

    data = pd.DataFrame(y)
    data[1] = u.mean(axis=1)
    data[2] = yspace
    data[3] = ubar.mean(axis=1)


    ax2.plot(data[1], data[0], color='blue', label='Priors')
    ax2.plot(data[3], data[2], color='red', label='Data')
    ax2.set_xlabel(r'$\overline{u}$', fontsize=20)
    ax2.set_ylabel(r'$y$', fontsize=20)

    plt.legend(fontsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.tick_params(axis='x', labelsize=20)
    plt.tight_layout()
    if not plot:
        plt.close()
    return u_pred

prior_wind_u = get_prior_wind(nn, x0, x1, y0, y1)


# In[17]:


def posterior_mean_field(kernel_size, bias_size, dtype=None):
    num = kernel_size + bias_size
    return tf.keras.Sequential([
        tfpl.VariableLayer(tfpl.IndependentNormal.params_size(num), dtype=dtype),
        tfpl.IndependentNormal(num)])

def bayesian_network(layers, kl_w, prior_means, prior_sd, phys_informed, num_inputs, num_outputs):
    
    def prior_distributions_dict(prior_means, prior_sd):
        priors_dict = {}

        for i, w in enumerate(prior_means):
            name = 'p' + str(i)
            def prior(kernel_size, bias_size, dtype=None, i=i, w=w):
                num = kernel_size + bias_size
                if phys_informed:
                    sd = tf.math.abs(w*prior_sd)
                    prior_model = tf.keras.Sequential([
                        tfp.layers.DistributionLambda(lambda t: tfd.Independent(tfd.Normal(loc=w, scale=sd/num), reinterpreted_batch_ndims=1)),])
                else:
                    sd = tf.convert_to_tensor([1.] * num)
                    prior_model = tf.keras.Sequential([ 
                        tfp.layers.DistributionLambda(lambda t: tfd.Independent(tfd.Normal(loc=0, scale=sd/num), reinterpreted_batch_ndims=1)),])
                return prior_model

            priors_dict[name] = prior

        return priors_dict

    inputs = tf.keras.layers.Input(shape=(num_inputs,))
    x = inputs
    
    my_priors = prior_distributions_dict(prior_means, prior_sd)
    keys = my_priors.keys()
    keys = [str(i) for i in keys]

    x = tfpl.DenseVariational(units=layers[0], input_shape=(num_inputs,), make_posterior_fn=posterior_mean_field, make_prior_fn=my_priors['p0'],
                                    kl_weight=1/kl_w, activation='tanh', kl_use_exact=True)(x) 
    
    for i, k in enumerate(keys[1:-1]):
        x = tfpl.DenseVariational(units=layers[i], make_posterior_fn=posterior_mean_field, make_prior_fn=my_priors[k],
                                    kl_weight=1/kl_w, activation='tanh', kl_use_exact=True)(x)
    
    x = tfpl.DenseVariational(units=num_outputs, make_posterior_fn=posterior_mean_field, make_prior_fn=my_priors[keys[-1]],
                                    kl_weight=1/kl_w, activation='tanh', kl_use_exact=True)(x) 
    
    distribution_params = tf.keras.layers.Dense(units=num_outputs*2, use_bias=False)(x)
    outputs = tfp.layers.IndependentNormal(num_outputs, validate_args=True)(distribution_params)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)


# In[18]:


def get_update_data(n_locs, pct_cs, seed=14, data=data, plot=False):

    np.random.seed(seed)

    n_y = int(pct_cs*len(data.y.unique()))
    x_dots = np.random.choice(data.x.unique(), n_locs, replace=False)
    data_bnn = pd.DataFrame()

    for i in range(n_locs):
        y_dots_temp = np.random.choice(np.arange(len(data.y.unique())), n_y, replace=False)
        data_temp = data[data.x == x_dots[i]].reset_index()
        data_temp = data_temp.loc[y_dots_temp]
        data_bnn = pd.concat([data_bnn, data_temp])

    data_bnn.reset_index(drop=True, inplace=True)

    xy_bnn = data_bnn[['x','y']].to_numpy() 
    target_bnn = data_bnn[['u','uv']].to_numpy() 
    target_bnn[..., 1] = -target_bnn[..., 1]
    
    fig, ax = plt.subplots(figsize=(8,3))
    im1 = ax.scatter(xy_bnn[:,0], xy_bnn[:,1], c=target_bnn[:,0], s=3, cmap='coolwarm')
    ax.set_title('Points chosen for Bayesian update (n = {})'.format(len(xy_bnn)))
    plt.colorbar(im1, ax=ax, label=r'$m/s$', orientation='horizontal', pad=0.15)
    if not plot:
        plt.close()

    return xy_bnn, target_bnn

# In[67]:


def get_posteriors(post_iters, bnn_pi, bnn, id=0):

    pred_mean_pi = np.zeros(shape=(post_iters, xy.shape[0], 1))
    pred_stdv_pi = np.zeros(shape=(post_iters, xy.shape[0], 1))
    pred_mean = np.zeros(shape=(post_iters, xy.shape[0], 1))
    pred_stdv = np.zeros(shape=(post_iters, xy.shape[0], 1))
    for i in range(post_iters):
        u_pred = bnn(xy)
        u_pred_pi = bnn_pi(xy)
        pred_mean[i] = u_pred.mean().numpy()[:,id].reshape(-1,1)
        pred_stdv[i] = u_pred.stddev().numpy()[:,id].reshape(-1,1)
        pred_mean_pi[i] = u_pred_pi.mean().numpy()[:,id].reshape(-1,1)
        pred_stdv_pi[i] = u_pred_pi.stddev().numpy()[:,id].reshape(-1,1) 

    return pred_mean_pi, pred_stdv_pi, pred_mean, pred_stdv


fmt = lambda x, pos: '{:.2f}'.format(x)

def plot_posteriors(pred_mean_pi, pred_stdv_pi, pred_mean, pred_stdv, plot, perc, data, prior_wind_u=prior_wind_u, xx=xx, yy=yy, y_prior=y_prior, u=u):

    zscore = st.norm.ppf(perc + (1-perc)/2)
    wind_u1d = u.mean(axis=2).mean(axis=1) 
    prior_wind_1d = prior_wind_u[:,0].reshape((xx.shape[0], xx.shape[1]), order='F').mean(axis=1)

    pred_mean_u_pi = pred_mean_pi.mean(axis=0).reshape(xx.shape, order='F')
    pred_mean_u = pred_mean.mean(axis=0).reshape(xx.shape, order='F')
    pred_stdv_u_pi = pred_stdv_pi.mean(axis=0).reshape(xx.shape, order='F')
    pred_stdv_u = pred_stdv.mean(axis=0).reshape(xx.shape, order='F')
    upper_u_pi = (pred_mean_u_pi + zscore*pred_stdv_u_pi).mean(axis=1)
    lower_u_pi = (pred_mean_u_pi - zscore*pred_stdv_u_pi).mean(axis=1)
    upper_u = (pred_mean_u + zscore*pred_stdv_u).mean(axis=1)
    lower_u = (pred_mean_u - zscore*pred_stdv_u).mean(axis=1)

    pred_mean_1d_pi = pred_mean_u_pi.mean(axis=1)
    pred_mean_1d = pred_mean_u.mean(axis=1)

    fig, ax = plt.subplots(2, 2, figsize=(20,15), gridspec_kw={'height_ratios': [1.5, 1]})
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    
    pred_means = np.stack([pred_mean_u_pi, pred_mean_u], axis=-1)
    pred_stdvs = np.stack([pred_stdv_u_pi, pred_stdv_u], axis=-1)

    
    vmin_mu = pred_means.min()
    vmax_mu = pred_means.max()
    vmin_sd = pred_stdvs.min()
    vmax_sd = pred_stdvs.max()

    data = data[:,:2] 

    cmap = cmr.get_sub_cmap('Reds', 0, .9)

    im = ax[0,0].contourf(xx, yy, pred_means[:,:,0], levels=100, cmap=cmap, vmin=vmin_mu, vmax=vmax_mu)
    ax[0,0].scatter(data[:,0], data[:,1], color='black', marker='x', label='data (n={:.0f})'.format(len(data)))
    ax[0,0].set_title('Physics-informed priors', fontsize=20)
    ax[0,0].set_ylabel(r'$y$ (distance from the wall, in $m$)', fontsize=18)
    ax[0,0].set_xlabel(r'$x$ (length of the PIV measurement window, in $m$)', fontsize=18)
    ax[0,0].annotate("(a)", xy=(-0.06, 1.05), xycoords="axes fraction", fontsize=18)


    im2 = ax[0,1].contourf(xx, yy, pred_means[:,:,1], levels=100, cmap=cmap, vmin=vmin_mu, vmax=vmax_mu)
    ax[0,1].set_title('Non-informed priors', fontsize=20)
    ax[0,1].scatter(data[:,0], data[:,1], color='black', marker='x', label='data (n={:.0f})'.format(len(data)))
    ax[0,1].set_ylabel(r'$y$ (distance from the wall, in $m$)', fontsize=18)
    ax[0,1].set_xlabel(r'$x$ (length of the PIV measurement window, in $m$)', fontsize=18)
    ax[0,1].annotate("(b)", xy=(1, 1.05), xycoords="axes fraction", fontsize=18)

    ax[0,0].tick_params(axis='y', labelsize=18)
    ax[0,0].tick_params(axis='x', labelsize=18)
    ax[0,1].tick_params(axis='y', labelsize=18)
    ax[0,1].tick_params(axis='x', labelsize=18)
    ax[0,1].yaxis.tick_right()
    ax[0,1].yaxis.set_label_position("right")

    norm = Normalize(vmin=vmin_mu, vmax=vmax_mu)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=[ax[0,0], ax[0,1]], orientation='horizontal', pad=0.15, aspect=15, shrink=0.6, format=FuncFormatter(fmt))
    #cb = plt.colorbar(im, ax=[ax[0, 0], ax[0, 1]], orientation='horizontal', pad=0.15, aspect=15, shrink=0.6, format=FuncFormatter(fmt))
    cb.set_ticks(np.linspace(vmin_mu, vmax_mu, 7))
    cb.set_label(label=r'$\overline{u}$ (in $m/s$)', size=20)
    cb.ax.tick_params(labelsize=20)

    ax[1,0].tick_params(axis='y', labelsize=18)
    ax[1,0].tick_params(axis='x', labelsize=18)
    ax[1,0].set_ylim(0.2,0.72)
    ax[1,0].set_xlim(-0.001,0.051)
    ax[1,0].set_yticks(np.round(np.linspace(0.25,0.7,6),2))
    ax[1,0].set_xticks(np.round(np.linspace(0,0.05,6),2))
    ax[1,0].plot(yy[:,0], wind_u1d, color='red', label='data')
    ax[1,0].plot(y_prior, prior_wind_1d, color='mediumblue', label='prior mean', alpha=0.5)
    ax[1,0].plot(yy[:,0], pred_mean_1d_pi, color='green', label='posterior mean')
    ax[1,0].fill_between(yy[:,0], upper_u_pi, lower_u_pi, color='gray', alpha=0.2, label='95% C.I.')
    ax[1,0].legend(loc='lower right', fontsize=18)
    ax[1,0].set_xlabel(r'$y$ (distance from the wall, in $m$)', fontsize=18)
    ax[1,0].set_ylabel(r'$\overline{u}$', fontsize=20)
    ax[1,0].annotate("(c)", xy=(-0.06, 1.05), xycoords="axes fraction", fontsize=18)

    ax[1,1].plot(yy[:,0], wind_u1d, color='red', label='data')
    ax[1,1].plot(yy[:,0], pred_mean_1d, color='green', label='posterior mean')
    ax[1,1].fill_between(yy[:, 0], upper_u, lower_u, color='gray', alpha=0.2, label='95% C.I.')
    ax[1,1].legend(loc='lower right', fontsize=18)
    ax[1,1].set_xlabel(r'$y$ (distance from the wall, in $m$)', fontsize=18)
    ax[1,1].set_ylabel(r'$\overline{u}$', fontsize=20)
    ax[1,1].annotate("(d)", xy=(1, 1.05), xycoords="axes fraction", fontsize=18)

    ax[1,1].tick_params(axis='y', labelsize=18)
    ax[1,1].tick_params(axis='x', labelsize=18)
    ax[1,1].yaxis.tick_right()
    ax[1,1].yaxis.set_label_position("right")
    ax[1,1].set_ylim(0.2,0.72)
    ax[1,1].set_xlim(-0.001,0.051)
    ax[1,1].set_xticks(np.round(np.linspace(0,0.05,6),2))
    ax[1,1].set_yticks(np.round(np.linspace(0.25,0.7,6),2))

    if not plot:
        plt.close()
        dict_pi = {'bias': ((pred_mean_u_pi-u.mean(axis=2))**2).mean(), 'var':pred_stdv_pi.mean(),'mse':((pred_mean_u_pi-u.mean(axis=2))**2).mean()+pred_stdv_pi.mean()}
        dict_npi = {'bias': ((pred_mean_u-u.mean(axis=2))**2).mean(), 'var':pred_stdv.mean(),'mse':((pred_mean_u-u.mean(axis=2))**2).mean()+pred_stdv.mean()}
        return {'pi': dict_pi, 'npi': dict_npi}

# In[20]:


def neg_loglikel(dist_true, dist_pred):
    return -dist_pred.log_prob(dist_true)

adam = optimizers.Adam(learning_rate=0.01)

epochs = 10000
batch_size = 4
prior_sd = 0.03

# COMPARING THIS WITH PRIOR VARIANCE IN SIMULATION STUDY
vars = np.zeros(len(prior_means))
for i, p in enumerate(prior_means):
    vars[i] = ((prior_sd*p / len(p))**2).mean()

print(vars.mean() / 0.0025**2)


# In[21]:


print('Starting Bayesian update...')
keras.utils.set_random_seed(812)

xy_bnn, target_bnn = get_update_data(n_locs=5, pct_cs=0.05, seed=1)
print('Using {} points.'.format(len(xy_bnn)))

kl_w = len(xy_bnn) / batch_size 

bnn_pi = bayesian_network(layers, kl_w, prior_means, prior_sd, phys_informed=True, num_inputs=2, num_outputs=2)
bnn_pi.compile(loss=neg_loglikel, optimizer=adam)
bnn_pi.fit(xy_bnn, target_bnn, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True);

bnn = bayesian_network(layers, kl_w, prior_means, prior_sd, phys_informed=False, num_inputs=2, num_outputs=2)
bnn.compile(loss=neg_loglikel, optimizer=adam)
bnn.fit(xy_bnn, target_bnn, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True);


# In[22]:


pred_mean_pi, pred_stdv_pi, pred_mean, pred_stdv = get_posteriors(100, bnn_pi, bnn)
base = {'mu_pi': pred_mean_pi, 'sd_pi': pred_stdv_pi, 'mu_npi': pred_mean, 'sd_npi':pred_stdv}
plot_posteriors(base['mu_pi'], base['sd_pi'], base['mu_npi'], base['sd_npi'], data=xy_bnn, plot=True, perc=.95)

# ## LOWER VARIANCE

# In[23]:


print('Starting Bayesian update...')
print('Using {} points.'.format(len(xy_bnn)))

keras.utils.set_random_seed(812)
prior_sd = 0.01 / 100

bnn_pi2 = bayesian_network(layers, kl_w, prior_means, prior_sd, phys_informed=True, num_inputs=2, num_outputs=2)
bnn_pi2.compile(loss=neg_loglikel, optimizer=adam)
bnn_pi2.fit(xy_bnn, target_bnn, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True);

bnn2 = bayesian_network(layers, kl_w, prior_means, prior_sd, phys_informed=False, num_inputs=2, num_outputs=2)
bnn2.compile(loss=neg_loglikel, optimizer=adam)
bnn2.fit(xy_bnn, target_bnn, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True);
print('Bayesian update complete.')


# In[24]:


pred_mean_pi2, pred_stdv_pi2, pred_mean2, pred_stdv2 = get_posteriors(100, bnn_pi2, bnn2)
base2 = {'mu_pi': pred_mean_pi2, 'sd_pi': pred_stdv_pi2, 'mu_npi': pred_mean2, 'sd_npi':pred_stdv2}
plot_posteriors(base2['mu_pi'], base2['sd_pi'], base2['mu_npi'], base2['sd_npi'], data=xy_bnn, plot=True, perc=.95)


dict_base = plot_posteriors(base['mu_pi'], base['sd_pi'], base['mu_npi'], base['sd_npi'], data=xy_bnn, plot=False, perc=.95)
dict_base2 = plot_posteriors(base2['mu_pi'], base2['sd_pi'], base2['mu_npi'], base2['sd_npi'], data=xy_bnn, plot=False, perc=.95)

print('metric: PI ; PI(low) ; NPI')
for i, k in enumerate(dict_base['pi'].keys()):
    if i < 3:
        print(str(k) + ': {:.2f} ; {:.2f}   ; {:.2f}'.format(1000*dict_base['pi'][k], 1000*dict_base2['pi'][k], 1000*dict_base['npi'][k]))
# 

# In[51]:


fmt = lambda x, pos: '{:.2f}'.format(x)
from matplotlib.colors import Normalize

def plot__uv_posteriors(uv_mean_pi, uv_mean, uvbar=uvbar, xx=xx, yy=yy, data=xy_bnn):

    pred_mean_u_pi = uv_mean_pi.mean(axis=0).reshape(uvbar.shape, order='F')
    pred_mean_u = uv_mean.mean(axis=0).reshape(uvbar.shape, order='F')

    fig, ax = plt.subplots(2, 2, figsize=(20,20))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    
    pred_means = 10000 * np.stack([pred_mean_u_pi, pred_mean_u], axis=-1) # -uvbar

    vmin_mu = pred_means.min()
    vmax_mu = pred_means.max()
    data = data[:,:2] 
    cmap = cmr.get_sub_cmap('Reds', 0, .7)
    #cmap = 'Reds'

    im = ax[0,0].contourf(xx, yy, pred_means[:,:,0], levels=100, cmap=cmap) #, vmin=vmin_mu, vmax=vmax_mu)
    ax[0,0].scatter(data[:,0], data[:,1], color='black', marker='x', label='data (n={:.0f})'.format(len(data)))
    ax[0,0].set_title('Physics-informed priors', fontsize=20)
    ax[0,0].set_ylabel(r'$y$ (distance from the wall, in $m$)', fontsize=18)
    ax[0,0].set_xlabel(r'$x$ (length of the PIV measurement window, in $m$)', fontsize=18)
    ax[0,0].annotate("(a)", xy=(-0.06, 1.05), xycoords="axes fraction", fontsize=18)

    cmap = cmr.get_sub_cmap('Reds', 0.8, 1)
    im1 = ax[0,1].contourf(xx, yy, pred_means[:,:,1], levels=100, cmap=cmap) #, vmin=vmin_mu, vmax=vmax_mu)
    ax[0,1].set_title('Non-informed priors', fontsize=20)
    ax[0,1].scatter(data[:,0], data[:,1], color='black', marker='x', label='data (n={:.0f})'.format(len(data)))
    ax[0,1].set_ylabel(r'$y$ (distance from the wall, in $m$)', fontsize=18)
    ax[0,1].set_xlabel(r'$x$ (length of the PIV measurement window, in $m$)', fontsize=18)
    ax[0,1].annotate("(b)", xy=(1, 1.05), xycoords="axes fraction", fontsize=18)

    ax[0,0].tick_params(axis='y', labelsize=18)
    ax[0,0].tick_params(axis='x', labelsize=18)
    ax[0,1].tick_params(axis='y', labelsize=18)
    ax[0,1].tick_params(axis='x', labelsize=18)
    ax[0,1].yaxis.tick_right()
    ax[0,1].yaxis.set_label_position("right")

    #norm = Normalize(vmin=vmin_mu, vmax=vmax_mu)

    cb = plt.colorbar(im, ax=ax[0,0], orientation='horizontal', pad=0.15, aspect=15, shrink=0.6, format=FuncFormatter(fmt))
    cb.set_ticks(10000*np.linspace(pred_mean_u_pi.min(), pred_mean_u_pi.max(), 5))
    cb.set_label(label=r"$\widehat{\overline{u'v'}}$ (in $10^{-4}$ $m/s$)", size=20)
    cb.ax.tick_params(labelsize=20)

    cb3 = plt.colorbar(im1, ax=ax[0,1], orientation='horizontal', pad=0.15, aspect=15, shrink=0.6, format=FuncFormatter(fmt))
    cb3.set_ticks(10000*np.linspace(pred_mean_u.min(), pred_mean_u.max(), 5))
    cb3.set_label(label=r"$\widehat{\overline{u'v'}}$ (in $10^{-4}$ $m/s$)", size=20)
    cb3.ax.tick_params(labelsize=20)

    pred_means = 10000 * np.stack([pred_mean_u_pi-uvbar, pred_mean_u-uvbar], axis=-1)
    pred_means = np.abs(pred_means)

    vmin_mu = pred_means.min()
    vmax_mu = pred_means.max() 

    cmap = 'Greys'

    im2 = ax[1,0].contourf(xx, yy, pred_means[:,:,0], levels=100, cmap=cmap, vmin=vmin_mu, vmax=vmax_mu)
    ax[1,0].scatter(data[:,0], data[:,1], color='black', marker='x', label='data (n={:.0f})'.format(len(data)))
    ax[1,0].set_ylabel(r'$y$ (distance from the wall, in $m$)', fontsize=18)
    ax[1,0].set_xlabel(r'$x$ (length of the PIV measurement window, in $m$)', fontsize=18)
    ax[1,0].annotate("(c)", xy=(-0.06, 1.05), xycoords="axes fraction", fontsize=18)


    im2 = ax[1,1].contourf(xx, yy, pred_means[:,:,1], levels=100, cmap=cmap, vmin=vmin_mu, vmax=vmax_mu)
    ax[1,1].scatter(data[:,0], data[:,1], color='black', marker='x', label='data (n={:.0f})'.format(len(data)))
    ax[1,1].set_ylabel(r'$y$ (distance from the wall, in $m$)', fontsize=18)
    ax[1,1].set_xlabel(r'$x$ (length of the PIV measurement window, in $m$)', fontsize=18)
    ax[1,1].annotate("(d)", xy=(1, 1.05), xycoords="axes fraction", fontsize=18)

    ax[1,0].tick_params(axis='y', labelsize=18)
    ax[1,0].tick_params(axis='x', labelsize=18)
    ax[1,1].tick_params(axis='y', labelsize=18)
    ax[1,1].tick_params(axis='x', labelsize=18)
    ax[1,1].yaxis.tick_right()
    ax[1,1].yaxis.set_label_position("right")

    norm = Normalize(vmin=vmin_mu, vmax=vmax_mu)

    cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=[ax[1,0], ax[1,1]], orientation='horizontal', pad=0.15, aspect=15, shrink=0.6, format=FuncFormatter(fmt))
    cb2.set_ticks(np.linspace(vmin_mu, vmax_mu, 7))
    cb2.set_label(label=r"$\left|\widehat{\overline{u'v'}}-\overline{u'v'}\right|$ $($in $10^{-4}$ $m/s)$", size=20)
    cb2.ax.tick_params(labelsize=20)


uv_mean_pi, _, uv_mean, _ = get_posteriors(100, bnn_pi, bnn, id=1)
plot__uv_posteriors(uv_mean_pi, uv_mean)

print(((uv_mean_pi.mean(axis=0).reshape(uvbar.shape, order='F') - uvbar)**2).mean())
print(((uv_mean.mean(axis=0).reshape(uvbar.shape, order='F') - uvbar)**2).mean())
