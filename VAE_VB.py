
# coding: utf-8

import torch
import numpy as np
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
# set training parameters
mnist = input_data.read_data_sets('data/mnist/', one_hot=True)
n_train_samples = mnist.train.num_examples
n_test_samples = mnist.test.num_examples
batch_size = 50
num_z_sample = 1000
num_epochs = 100
total_batches = int(n_train_samples/batch_size)
total_batches_test = int(n_test_samples/batch_size)
dim_z = 2
dim_h = 512
dim_x = mnist.train.images.shape[1]
learning_rate = 1e-4
init_std = 0.1 
log2pi = 1.83787706641

# define bottom-up recognition network q(z|x)
W_xh = Variable(torch.randn(dim_x, dim_h)*init_std, requires_grad=True)
b_xh = Variable(torch.zeros(dim_h), requires_grad=True)
W_hz_mu = Variable(torch.randn(dim_h, dim_z)*init_std, requires_grad=True)
b_hz_mu = Variable(torch.zeros(dim_z), requires_grad=True)
W_hz_sigma = Variable(torch.randn(dim_h, dim_z)*init_std, requires_grad=True)
b_hz_sigma = Variable(torch.zeros(dim_z), requires_grad=True)

def Q(x):
    # output z_mu and z_sigma, and samples from the Gaussian
    # all with size (batchsize, dim_z)
    h = x.mm(W_xh).add(b_xh.repeat(x.size(0),1)).clamp(min=0)
    z_mu = h.mm(W_hz_mu).add(b_hz_mu.repeat(h.size(0),1))
    z_log_var = h.mm(W_hz_sigma).add(b_hz_sigma.repeat(h.size(0),1))
    eps = Variable(torch.randn(z_mu.size(0), z_mu.size(1)))
    z_sample = z_mu + z_log_var.exp().sqrt() * eps
    return z_mu, z_log_var, z_sample

def Q_multi_z(z_mu, z_log_var, num_z_sample):
    # return z_sample as size (num_z_sample, batchsize, dim_z)
    eps = Variable(torch.randn(num_z_sample, z_mu.size(0), z_mu.size(1)))
    z_sample = (z_mu.unsqueeze(0).expand(num_z_sample,*z_mu.size())
                + z_log_var.exp().sqrt().unsqueeze(0).expand(num_z_sample,*z_log_var.size()) * eps)
    return z_sample

# define top-down generator network p(x|z)
W_zh = Variable(torch.randn(dim_z, dim_h)*init_std, requires_grad=True)
b_zh = Variable(torch.zeros(dim_h), requires_grad=True)
W_hx = Variable(torch.randn(dim_h, dim_x)*init_std, requires_grad=True)
b_hx = Variable(torch.zeros(dim_x), requires_grad=True)

def batch_mat_mult(X, Y):
    # batch matrix multiplication Z(b,m,p)=X(b,m,n)*Y(n,p)
    return torch.bmm(X, Y.unsqueeze(0).expand(X.size(0), *Y.size()))

def P(z):
    # top-down generator of samples of x, size (batchsize, dim_x)
    h = z.mm(W_zh).add(b_zh.repeat(z.size(0),1)).clamp(min=0)
    x = torch.sigmoid(h.mm(W_hx).add(b_hx.repeat(h.size(0),1)))
    return x

def P_multi_x(z, num_z_sample):
    # generate x as size (num_z_sample, batchsize, dim_x)
    h = torch.clamp(batch_mat_mult(z, W_zh) + b_zh.repeat(z.size(0),z.size(1),1), min=0)
    x = torch.sigmoid(batch_mat_mult(h, W_hx) + b_hx.repeat(h.size(0),h.size(1),1))
    return x

def log_sum_exp(A):
    ma,_ = torch.max(A,0)
    ma_rep = ma.expand(A.size(0),ma.size(1))
    lse = ma + torch.log(torch.sum(torch.exp(A-ma_rep),0))
    return lse

# define cost function and optimizer
parameters = [W_xh, b_xh, W_hz_mu, b_hz_mu, W_hz_sigma, b_hz_sigma, W_zh, b_zh, W_hx, b_hx]
optimizer = torch.optim.Adam(parameters, lr=learning_rate)

def loss(z_sample, z_mu, z_log_var, x_gen, x):
    KL = -torch.mean(0.5*torch.sum(1.0 + z_log_var - z_mu**2 - z_log_var.exp(), 1))
    LL = torch.mean(torch.sum(x * torch.log(x_gen + 1e-10) + (1-x)*torch.log(1-x_gen + 1e-10), 1))
    L = -LL + KL
    return L

def lower_bound(z_sample, z_mu, z_log_var, x_gen, x):
    x_bat = x.unsqueeze(0).expand(x_gen.size(0),*x.size())
    # compute log p(x|z)
    p_xz = torch.squeeze(torch.sum(x_bat * torch.log(x_gen + 1e-10) + (1-x_bat)*torch.log(1-x_gen + 1e-10), 2))
    # compute log q(z)
    ll = -0.5*log2pi - torch.pow(z_sample,2)/2
    p_z = 0
    for i in range(ll.size(2)):
        p_z = p_z + ll[:,:,i]
    # compute log q(z|x)
    z_mu_bat = z_mu.unsqueeze(0).expand(z_sample.size(0),*z_mu.size())
    z_var_bat = z_log_var.exp().unsqueeze(0).expand(z_sample.size(0),*z_log_var.size())
    z_log_var_bat = z_log_var.unsqueeze(0).expand(z_sample.size(0),*z_log_var.size())
    ll = -0.5*log2pi - 0.5*z_log_var_bat - torch.pow(z_sample-z_mu_bat,2)/z_var_bat/2
    q_zx = 0
    for i in range(ll.size(2)):
        q_zx = q_zx + ll[:,:,i]
    LB = torch.mean(log_sum_exp(p_xz + p_z - q_zx - np.log(p_xz.size(0))), 1)
    return torch.squeeze(LB)

# training with AEVB
d = 0
LB_train = Variable(torch.zeros(num_epochs,total_batches))
LB_test = Variable(torch.zeros(num_epochs,total_batches))
for epoch in range(num_epochs):
    for mb in range(total_batches):
        # get mini-batch
        mini_batch, _ = mnist.train.next_batch(batch_size)
        x = Variable(torch.from_numpy(mini_batch))
        # recognition
        z_mu, z_log_var, z_sample = Q(x)
        # generation
        x_gen = P(z_sample)
        # update
        L = loss(z_sample, z_mu, z_log_var, x_gen, x)
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        if mb % 100 == 0:
            print('Epoch: {}; Batch: {}; ELBO: {:.8}'.format(epoch, mb, -L.data[0]))
            
    # evaluate tigher lower bound of log(p(x))        
    if epoch % 10 == 0:
        print('evaluating lower bound...')
        # train set lower bound
        for mb in range(total_batches):
            # get mini-batch
            mini_batch, _ = mnist.train.next_batch(batch_size)
            x = Variable(torch.from_numpy(mini_batch))
            z_mu, z_log_var, _ = Q(x)
            # batch sample 1000 z's
            z_multi_sample_train = Q_multi_z(z_mu, z_log_var, num_z_sample)
            x_multi_sample_train = P_multi_x(z_multi_sample_train, num_z_sample)
            LB = lower_bound(z_multi_sample_train, z_mu, z_log_var, x_multi_sample_train, x)
            LB_train[mb,d] = LB
            print('Epoch: {}; Batch: {}/{}; L_train_1000: {:.8}'.format(epoch, mb, total_batches, LB.data[0]))
        # test set lower bound
        for mb in range(total_batches_test):
            # get mini-batch
            mini_batch_test, _ = mnist.test.next_batch(batch_size)
            x_test = Variable(torch.from_numpy(mini_batch_test))
            z_mu_test, z_log_var_test, _ = Q(x_test)
            # batch sample 1000 z's
            z_multi_sample_test = Q_multi_z(z_mu_test, z_log_var_test, num_z_sample)
            x_multi_sample_test = P_multi_x(z_multi_sample_test, num_z_sample)
            LB = lower_bound(z_multi_sample_test, z_mu_test, z_log_var_test, x_multi_sample_test, x_test)
            LB_test[mb,d] = LB
            print('Epoch: {}; Batch: {}/{}; L_test_1000: {:.8}'.format(epoch, total_batches_test, LB.data[0]))
        d = d + 1

    # plot figure to monitor training process
    samples = x_gen.data.numpy()[:100]
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    if not os.path.exists('out/'):
        os.makedirs('out/')
    plt.savefig('out/AEVB_Ep{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)

# plot average lower bound
LB_train_avg = torch.mean(LB_train,0)
LB_test_avg = torch.mean(LB_test,0)
epc = Variable(torch.FloatTensor([0.,10.,20.,30.,40.,50.,60.,70.,80.,90.]))
fg = plt.figure(figsize=(8,5))
plt.plot(epc.data.numpy().astype("float32"),LB_train_avg.data.numpy().astype("float32"),'r')
plt.plot(epc.data.numpy().astype("float32"),LB_test_avg.data.numpy().astype("float32"),'r--')
plt.grid()
plt.legend(['WS train'], loc='lower right')
plt.ylabel('Lower bound')
plt.xlabel('# of epochs')
plt.savefig('out/AEWS_LB1000.png', bbox_inches='tight')
plt.close(fg)

# plot 100 generated digits
z_sample_test = Variable(torch.randn(100,dim_z))
x_gen = P(z_sample_test)
samples = x_gen.data.numpy()[:100]
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(10, 10)
gs.update(wspace=0.05, hspace=0.05)
for i, sample in enumerate(samples):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
plt.savefig('out/AEVB_sample100.png', bbox_inches='tight')
plt.close(fig)

# plot latent space z
x, y_sample = mnist.test.next_batch(n_test_samples)
x_sample = Variable(torch.from_numpy(x))
z_mu, _, _ = Q(x_sample)
y_label = np.argmax(y_sample, 1)
f=plt.figure(figsize=(8, 6))
plt.scatter(z_mu.data.numpy()[:, 0], z_mu.data.numpy()[:, 1], c=y_label.astype("float32"), cmap='rainbow')
plt.colorbar()
plt.grid()
plt.savefig('out/AEVB_Latent.png', bbox_inches='tight')
plt.close(f)

# plot grid visualization in space z
nx = ny = 20
x_values = torch.linspace(-3, 3, steps=20)
y_values = torch.linspace(-3, 3, steps=20)
canvas = np.empty((28*ny, 28*nx))
for i, yi in enumerate(x_values):
    for j, xi in enumerate(y_values):
        z_mu = Variable(torch.FloatTensor([[xi,yi]]))
        z_mu = z_mu.repeat(batch_size,1)
        x_mean = P(z_mu)
        canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].data.numpy().reshape(28, 28)
ff=plt.figure(figsize=(8, 10))
Xi, Yi = np.meshgrid(x_values, y_values)
plt.imshow(canvas, origin="upper", cmap="gray")
plt.tight_layout()
plt.grid()
plt.axis('off')
plt.savefig('out/AEVB_Visualize_Latent_Grid.png', bbox_inches='tight')
plt.close(ff)

