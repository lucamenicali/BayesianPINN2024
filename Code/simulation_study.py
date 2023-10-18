import pickle
import os 
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize
from matplotlib.gridspec import GridSpec
import tensorflow_probability as tfp
import numpy as np
import math
import time
import datetime
from datetime import timedelta
import random
import keras
import pandas as pd
from keras import optimizers
from tensorflow.keras.layers import concatenate

tfpl = tfp.layers
tfd = tfp.distributions


# ## Prior mean calibration

# In[5]:


class Network:
    def __init__(self, layers):
        self.layers = layers

    def get_network(self, num_inputs=2, activation='tanh', num_outputs=1):
        
        inputs = tf.keras.layers.Input(shape=(num_inputs,))
        x = inputs
        for l in self.layers:
            x = tf.keras.layers.Dense(l, activation=activation)(x)
        outputs = tf.keras.layers.Dense(num_outputs)(x)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs)


# In[7]:


class GradientLayer(tf.keras.layers.Layer):
    def __init__(self,model,**kwargs):
        self.model=model
        super().__init__(**kwargs)
    def call(self, xt):
        with tf.GradientTape() as g:
            g.watch(xt) 
            with tf.GradientTape() as gg: 
                gg.watch(xt)
                u = self.model(xt)
                du_dxt = gg.batch_jacobian(u, xt) 
                du_dt = du_dxt[..., 1]  
                du_dx = du_dxt[..., 0]
        d2u_dx2 = g.batch_jacobian(du_dx, xt)[..., 0]
        return u, du_dt, du_dx, d2u_dx2

class PINN:
    def __init__(self, network, nu, cosine):
        self.network = network
        self.nu = nu
        self.cosine = cosine
        self.grads = GradientLayer(self.network)
    def build(self):

        xt_eqn = tf.keras.layers.Input(shape=(2,))
        xt_ini = tf.keras.layers.Input(shape=(2,))
        if self.cosine:
            xt_left = tf.keras.layers.Input(shape=(2,))
            xt_right = tf.keras.layers.Input(shape=(2,))
        else:
            xt_bnd = tf.keras.layers.Input(shape=(2,))

        # compute gradients. we use the gradient class previously built.
        u, du_dt, du_dx, d2u_dx2 = self.grads(xt_eqn)

        # equation output being zero
        u_eqn = du_dt + u*du_dx - (self.nu)*d2u_dx2
        u_ini = self.network(xt_ini) 
        if self.cosine:
            u_left = self.network(xt_left)
            u_right = self.network(xt_right)
            u_bnd = u_left - u_right
            return tf.keras.models.Model(inputs=[xt_eqn, xt_ini, xt_left, xt_right], outputs=[u_eqn, u_ini, u_bnd, u_bnd])
        else:
            u_bnd = self.network(xt_bnd)
            return tf.keras.models.Model(inputs=[xt_eqn, xt_ini, xt_bnd], outputs=[u_eqn, u_ini, u_bnd])


# In[8]:


class L_BFGS_B:

    def __init__(self, model, x_train, u_train, maxiter, factr=1e7, m=50, maxls=50):

        # set attributes
        self.model = model
        self.x_train = [ tf.constant(x, dtype=tf.float32) for x in x_train ]
        self.u_train = [ tf.constant(u, dtype=tf.float32) for u in u_train ]
        self.factr = factr
        self.m = m
        self.maxls = maxls
        self.maxiter = maxiter
        self.metrics = ['loss']
        # initialize the progress bar
        self.progbar = tf.keras.callbacks.ProgbarLogger(
            count_mode='steps', stateful_metrics=self.metrics)
        self.progbar.set_params( {
            'verbose':1, 'epochs':1, 'steps':self.maxiter, 'metrics':self.metrics})

    def set_weights(self, flat_weights):
        
        # get model weights
        shapes = [ w.shape for w in self.model.get_weights() ]
        # compute splitting indices
        split_ids = np.cumsum([ np.prod(shape) for shape in [0] + shapes ])
        # reshape weights
        weights = [ flat_weights[from_id:to_id].reshape(shape)
            for from_id, to_id, shape in zip(split_ids[:-1], split_ids[1:], shapes) ]
        # set weights to the model
        self.model.set_weights(weights)

    @tf.function
    def tf_evaluate(self, x, u):
        with tf.GradientTape() as g:
            loss = tf.reduce_mean(tf.keras.losses.mse(self.model(x), u))
        grads = g.gradient(loss, self.model.trainable_variables)
        return loss, grads

    def evaluate(self, weights):

        # update weights
        self.set_weights(weights)
        # compute loss and gradients for weights
        loss, grads = self.tf_evaluate(self.x_train, self.u_train)
        # convert tf.Tensor to flatten ndarray
        loss = loss.numpy().astype('float64')
        grads = np.concatenate([ g.numpy().flatten() for g in grads ]).astype('float64')

        return loss, grads

    def callback(self, weights):

        self.progbar.on_batch_begin(0)
        loss, _ = self.evaluate(weights)
        self.progbar.on_batch_end(0, logs=dict(zip(self.metrics, [loss])))

    def fit(self):

        # get initial weights as a flat vector
        initial_weights = np.concatenate(
            [ w.flatten() for w in self.model.get_weights() ])
        # optimize the weight vector
        print('Optimizer: L-BFGS-B (maxiter={})'.format(self.maxiter))
        self.progbar.on_train_begin()
        self.progbar.on_epoch_begin(1)
        scipy.optimize.fmin_l_bfgs_b(func=self.evaluate, x0=initial_weights,
            factr=self.factr, m=self.m, maxls=self.maxls, maxiter=self.maxiter,
            callback=self.callback)
        self.progbar.on_epoch_end(1)
        self.progbar.on_train_end()


# In[9]:


class TrainingData:
    def __init__(self, n_init, n_bnd, cosine):
        self.n_init = n_init
        self.n_bnd = n_bnd
        self.cosine = cosine

    def get_training_data(self):

        xt_eqn = np.array(np.meshgrid(np.linspace(-1, 1, self.n_bnd), np.linspace(0, 2, self.n_init))).T.reshape(-1,2)

        xt_ini = np.array(np.meshgrid(np.linspace(-1, 1, self.n_bnd), np.array([0]))).T.reshape(-1,2)
        xt_ini = np.tile(xt_ini, (self.n_init, 1))

        u_eqn = np.zeros((self.n_init*self.n_bnd,1))
        u_bnd = u_eqn

        if self.cosine:
            xt_left = np.array(np.meshgrid(np.array([-1]), np.linspace(0, 2, self.n_init))).T.reshape(-1,2)
            xt_left = np.tile(xt_left, (self.n_bnd, 1))
            xt_right = np.array(np.meshgrid(np.array([1]), np.linspace(0, 2, self.n_init))).T.reshape(-1,2)
            xt_right = np.tile(xt_right, (self.n_bnd, 1))
            u_ini = np.cosine(np.pi * xt_ini[..., 0, np.newaxis])

            x_train = [xt_eqn, xt_ini, xt_left, xt_right] 
            u_train = [u_eqn, u_ini, u_bnd, u_bnd]
        else:
            xt_bnd = np.array(np.meshgrid(np.array([-1,1]), np.linspace(0, 2, self.n_init//2))).T.reshape(-1,2)
            xt_bnd = np.tile(xt_bnd, (self.n_bnd, 1))
            u_eqn = np.zeros((self.n_init*self.n_bnd, 1))
            u_ini = np.sin(-np.pi * xt_ini[..., 0, np.newaxis])

            x_train = [xt_eqn, xt_ini, xt_bnd] 
            u_train = [u_eqn, u_ini, u_bnd]

        return {'x_train': x_train, 'u_train': u_train}


# In[14]:


def prior_training(layers, n_init, n_bnd, nu, u_true, cosine, maxiter):

    keras.utils.set_random_seed(1)

    network = Network(layers)
    NN = network.get_network()

    pinn = PINN(network=NN, nu=nu, cosine=cosine).build()
    training_data = TrainingData(n_init=n_init, n_bnd=n_bnd, cosine=cosine).get_training_data()
    lbfgs = L_BFGS_B(model=pinn, x_train=training_data['x_train'], u_train=training_data['u_train'], maxiter=maxiter)
    lbfgs.fit()

    weights = []
    for l in range(1, len(layers)+2):
        temp = np.append(NN.layers[l].get_weights()[0].flatten(), NN.layers[l].get_weights()[1].flatten())
        weights.append(temp)

    return weights


# In[40]:


layers = [10,10,10,10]
n_init = 1000
n_bnd = 100
nu = 0.05
cosine = False
maxiter = 150000

u_base = np.loadtxt('BurgersEquationBase.txt')
#u_low_nu = np.loadtxt('BurgersEquationLowViscosity.txt')
#u_cosine = np.loadtxt('BurgersEquationCosine.txt')

u_true = u_base

prior_means = prior_training(layers=layers, n_init=n_init, n_bnd=n_bnd, nu=nu, u_true=u_true, cosine=cosine, maxiter=maxiter)


# ## Bayesian Update

# In[41]:
def posterior(kernel_size, bias_size, dtype=None):
    num = kernel_size + bias_size
    return tf.keras.Sequential([
        tfpl.VariableLayer(tfpl.IndependentNormal.params_size(num), dtype=dtype),
        tfpl.IndependentNormal(num)])



def bayesian_network(layers, kl_w, prior_means, prior_sd, phys_informed, num_inputs=2, num_outputs=1):
    
    def prior_distributions_dict(prior_means, prior_sd):
        priors_dict = {}

        for i, w in enumerate(prior_means):
            name = 'p' + str(i)
            def prior(kernel_size, bias_size, dtype=None, i=i, w=w):
                num = kernel_size + bias_size
                if phys_informed:
                    sd = tf.convert_to_tensor([prior_sd] * num)
                    prior_model = tf.keras.Sequential([
                        tfp.layers.DistributionLambda(lambda t: tfd.Independent(tfd.Normal(loc=w, scale=sd), reinterpreted_batch_ndims=1)),])
                else:
                    sd = tf.convert_to_tensor([1.] * num)
                    prior_model = tf.keras.Sequential([ 
                        tfp.layers.DistributionLambda(lambda t: tfd.Independent(tfd.Normal(loc=0, scale=sd), reinterpreted_batch_ndims=1)),])
                return prior_model

            priors_dict[name] = prior

        return priors_dict

    inputs = tf.keras.layers.Input(shape=(num_inputs,))
    x = inputs
    
    my_priors = prior_distributions_dict(prior_means, prior_sd)
    keys = my_priors.keys()
    keys = [str(i) for i in keys]

    x = tfpl.DenseVariational(units=layers[0], input_shape=(num_inputs,), make_posterior_fn=posterior, make_prior_fn=my_priors['p0'],
                                    kl_weight=1/kl_w, activation='tanh')(x) 
    
    for i, k in enumerate(keys[1:-1]):
        x = tfpl.DenseVariational(units=layers[i], make_posterior_fn=posterior, make_prior_fn=my_priors[k],
                                    kl_weight=1/kl_w, activation='tanh')(x)
    
    x = tfpl.DenseVariational(units=num_outputs, make_posterior_fn=posterior, make_prior_fn=my_priors[keys[-1]],
                                    kl_weight=1/kl_w, activation='tanh')(x) 
    
    distribution_params = tf.keras.layers.Dense(units=num_outputs*2)(x)
    outputs = tfp.layers.IndependentNormal(num_outputs, validate_args=True)(distribution_params)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)


# In[30]:


def negative_loglikelihood(true_dist, estimated_dist):
    return -estimated_dist.log_prob(true_dist)

def bayesian_update(n_update, batch_size, prior_means, prior_sd, lr, iters, physics_noise, u_true, seed, phys_informed, t0, t1, x_list_id):
    start_time = time.time()

    kl_w = n_update / batch_size
    
    bnn = bayesian_network(layers, kl_w, prior_means, prior_sd, phys_informed)
    
    # adding noise to the true solution
    np.random.seed(seed)
    noise = np.random.normal(0, physics_noise, u_true.shape[0]*u_true.shape[1])
    noise = noise.reshape((u_true.shape[0], u_true.shape[1]))
    u_noisy = np.sum([u_true, noise], axis=0).flatten(order='F')
    
    # creating complete grid
    tspace = np.linspace(0, 2, u_true.shape[0])
    xspace = np.linspace(-1, 1, u_true.shape[1])
    XTgrid = np.array(np.meshgrid(xspace, tspace)).T.reshape(-1,2)
    XTgrid = pd.DataFrame(XTgrid, columns=['x','t'])
    x_list = xspace[x_list_id]
    XTgrid_restr = XTgrid[(XTgrid.t >= t0) & (XTgrid.t <= t1) & (XTgrid.x.isin(x_list))]
    u_noisy_restr = pd.DataFrame(u_noisy).loc[XTgrid_restr.index]

    random.seed(seed)
    update_ids = random.sample(list(XTgrid_restr.index), n_update)
    xt_bnn = XTgrid_restr.loc[update_ids]
    u_bnn = u_noisy_restr.loc[update_ids]

    print('Starting Bayesian update...')
    keras.utils.set_random_seed(seed)
    bnn.compile(loss=negative_loglikelihood, optimizer=optimizers.Adam(learning_rate=lr))
    bnn.fit(xt_bnn, u_bnn, epochs=iters, batch_size=batch_size, verbose=0, shuffle=True);
    print('Update done.')
    print()

    print('Calculating posterior...')
    post_iters = 100
    pred_mean = np.zeros(shape=(post_iters, u_true.shape[0]*u_true.shape[1]))
    pred_stdv = np.zeros(shape=(post_iters, u_true.shape[0]*u_true.shape[1]))
    for i in range(post_iters):
        prediction_distribution = bnn(XTgrid.to_numpy())
        pred_mean[i] = prediction_distribution.mean().numpy().flatten()
        pred_stdv[i] = prediction_distribution.stddev().numpy().flatten()
    print('Calculations done.')

    upper = pred_mean + 1.96 * pred_stdv
    lower = pred_mean - 1.96 * pred_stdv
    upper = np.mean(upper, axis=0)
    lower = np.mean(lower, axis=0)
    prediction_mean = np.mean(pred_mean, axis=0)
    prediction_sd = np.mean(pred_stdv, axis=0)

    mse = ((prediction_mean-u_true.flatten(order='F'))**2).mean()
    
    df_bnn = pd.DataFrame(np.array(u_bnn)[:,0], columns=['u_bnn'], index=u_bnn.index)
    df_bnn[['x','t']] = xt_bnn

    runtime = round(time.time() - start_time,0)
    
    summary_dict = {
        'n_update': n_update,
        'prior_sd': prior_sd,
        'lr': lr,
        'iters': iters,
        'physics_noise': physics_noise,
        'seed': seed,
        'phys_informed': phys_informed,
        'mse': mse,
        'prediction_mean': prediction_mean,
        'prediction_sd': prediction_sd,
        'df_bnn': df_bnn,
        'runtime': runtime
    }  
    
    return summary_dict


# In[55]:


n_update = 50
batch_size = 4

prior_sd = 0.05**2
lr = 0.001
iters = 10000
physics_noise = 0.1

seed = 1
phys_informed = True
t0 = 0
t1 = 2
x_list_id = np.linspace(0,99,100).astype('int')

random.seed(seed)
np.random.seed(seed)

dict_seeds = {}


for seed in range(1,2):
    random.seed(seed)
    np.random.seed(seed) 
    keras.utils.set_random_seed(seed)

    phys_informed = True
    temp_pi = bayesian_update(n_update, batch_size, prior_means, prior_sd, lr, iters, physics_noise, u_true, seed, phys_informed, t0, t1, x_list_id)

    phys_informed = False
    temp_npi = bayesian_update(n_update, batch_size, prior_means, prior_sd, lr, iters, physics_noise, u_true, seed, phys_informed, t0, t1, x_list_id)
    dict_seeds[f'seed_{seed}'] = {'pi': temp_pi, 'npi': temp_npi}

with open('dict_seeds.pkl', 'wb') as f:
    pickle.dump(dict_seeds, f)

