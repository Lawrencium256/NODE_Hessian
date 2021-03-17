#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic(u'capture', u'', u'"""\n-----------------------------------------------------------------------------------------------------------------------------------------------------------\nThis file is used to compute the hessian during training for fitting a simple 1D ODE using a normal neural network.\n-----------------------------------------------------------------------------------------------------------------------------------------------------------\n"""')


# In[2]:


get_ipython().run_cell_magic(u'capture', u'', u'%%bash \npip install torchdiffeq')


# In[3]:


import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# In[4]:


parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--manual_hessian', action='store_true')
parser.add_argument('--library_hessian', action='store_true')
parser.add_argument('--hessian_freq', type=int, default=20)
args = parser.parse_args(args=[])

args.batch_size = 10
args.batch_time = 20
args.niters=5000
args.test_freq=100
args.library_hessian = True
args.manual_hessian = False
args.viz = True
args.hessian_freq = 100
args.method = 'dopri5'


# In[5]:


#The technique only works when the adjoint method is not used. If it is used, the Hessian returned is a matrix of zeros.
adjoint = False

if adjoint == True:
    from torchdiffeq import odeint_adjoint as odeint
if adjoint == False:
    from torchdiffeq import odeint


# In[6]:


device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


# In[7]:


args.data_size = 1000

true_y0 = torch.tensor([2.]).to(device)
t_0, t_1 = 0., 2.
t = torch.linspace(t_0, t_1, args.data_size).to(device)


# In[8]:


class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.exp(t)


# In[9]:


#The true solution defines an exponential.
with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method = args.method)


# In[10]:


def get_batch():

    #List of 20 random integers.
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False)) 

    #List of 20 random points in the actual solution.
    batch_y0 = true_y[s]  # (M, D)  

    #The first (10) values from t.
    batch_t = t[:args.batch_time]  # (T)     
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)    Set of 20 lots of 10 sequential points in true_y.
    t_list = torch.stack([t[s + i] for i in range(args.batch_time)], dim=0)

    return batch_y0.to(device), batch_t.to(device), batch_y.to(device), t_list.to(device)


# In[11]:


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


# In[12]:


get_ipython().run_cell_magic(u'capture', u'', u"if args.viz:\n    makedirs('png')\n    import matplotlib.pyplot as plt\n    fig = plt.figure(figsize=(12, 4), facecolor='white')    ")


# In[13]:


def visualize(true_y, pred_y, odefunc, itr):

  """
  This slightly altered version of the function visualize() seems to work fine. The only change is that I have moved the plt.figure() part of the code
  inside the function itself, i.e. I am creating a new figure environment for every figure, instead of editing the same environment multiple times.
  """

  if args.viz:

    fig = plt.figure(figsize=(12, 4), facecolor='white')  #facecolor is the background colour.
    plt.plot(t.cpu().numpy(), true_y.cpu().numpy(), 'g-', label='True_y')
    plt.plot(t.cpu().numpy(), pred_y.cpu().detach().numpy(), 'b--', label='Predicted y')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    
    #plt.savefig('png/{:03d}'.format(itr))
    plt.draw()
    plt.pause(0.001)
    plt.close()


# In[14]:


class Ordinary_Net(nn.Module):

    def __init__(self):
        super(Ordinary_Net, self).__init__()

        #Define a very simple neural network architecture with 1 hidden layer.
        self.net = nn.Sequential(
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, 1),
        )
        
        for m in self.net.modules():
          if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.1)
            nn.init.constant_(m.bias, val=0)

    def forward(self, y):
        return self.net(y)

net = Ordinary_Net()


# In[15]:


def get_pred_y(t_array, network):
  """
  Evaluates network(t) for every t in an array of test values.
  """
  if t_array.ndim == 1:
    pred_y = torch.zeros_like(t_array).to(device)
    for i in range(len(t)):
      t_value = t_array[i]
      t_value = torch.reshape(t_value, (1,))
      input = t_value.to(device)
      output = network(input)
      pred_y[i] = output
    pred_y = torch.reshape(pred_y, (args.data_size, 1))

  else:  
    pred_y = torch.zeros_like(t_array).to(device)
    for i in range(t_array.shape[0]):
      for j in range(t_array.shape[1]):
        t_value = t_array[i,j]
        t_value = torch.reshape(t_value, (1,))
        input = t_value.to(device)
        output = network(input)
        pred_y[i,j] = output
    pred_y = torch.reshape(pred_y, (args.batch_time, args.batch_size, 1))
  
  return(pred_y)


# In[16]:


class Network(nn.Module):

  def __init__(self, a, b, c, d):
    super(Network, self).__init__()
    self.a = a
    self.b = b
    self.c = c
    self.d = d

  def forward(self, y):
    x = F.linear(y, self.a, self.b)
    m = nn.Tanh()
    x = m(x)
    x = F.linear(x, self.c, self.d)
    return x


def get_loss_square(params_vector):

  a = params_vector[:10].reshape([10, 1])
  b = params_vector[10:20].reshape([10])
  c = params_vector[20:30].reshape([1, 10])
  d = params_vector[30:31].reshape([1])
  
  neural_net = Network(a, b, c, d).to(device)
  pred_y = get_pred_y(t, neural_net).to(device) 
  loss = torch.mean(torch.abs(pred_y - true_y))
  return loss

def get_library_hessian(net):

  param_tensors = net.parameters()
  params_vector = torch.tensor([]).to(device)
  for param in param_tensors:
    vec = torch.reshape(param, (-1,)).to(device)
    params_vector = torch.cat((params_vector, vec))

  hessian = torch.autograd.functional.hessian(get_loss_square, params_vector)
  return hessian


# In[17]:


def get_manual_hessian(grads, parameters):
  """
  Calculation of the Hessian using nested for loops.
  Inputs: 
    grads:      tuple of gradient tensors. Created using something like grads = torch.autograd.grad(loss, parameters, create_graph=True).
    parameters: list of parameter objects. Created using something like parameters = optimizer.param_groups[0]['params'].
  """
  start = time.time()                       #Begin timer.

  n_params = 0
  for param in parameters:
    n_params += torch.numel(param)
  grads2 = torch.zeros(n_params,n_params)             #Create an matrix of zeros thas has the same shape as the Hessian.

  y_counter = 0                             #y_direction refers to row number in the Hessian.

  for grad in grads:
      grad = torch.reshape(grad, [-1])                                  #Rearrange the gradient information into a vector.        

      for j, g in enumerate(grad):
        x_counter = 0                                                   #x_direction refers to column number in the Hessian.

        for l, param in enumerate(parameters):
          g2 = torch.autograd.grad(g, param, retain_graph=True)[0]      #Calculate the gradient of an element of the gradient wrt one layer's parameters.
          g2 = torch.reshape(g2, [-1])                                  #Reshape this into a vector.
          len = g2.shape[0]                       
          grads2[j+y_counter, x_counter:x_counter+len] = g2             #Indexing ensures that the second order derivatives are placed in the correct positions.
          x_counter += len

      grads2 = grads2.to(device)
      y_counter += grad.shape[0]
      print("Gradients calculated for row number " + str(y_counter) + ".")
  
  print('Time used was ', time.time() - start)

  return grads2


# In[ ]:


if __name__ == '__main__':
    """
    Executes the programme. This includes doing the following:

      - Trains the network;
      - Outputs the results in a series of png files (if desired);
      - Outputs hessian matrix information in list form.
    """

    ii = 0

    net = Ordinary_Net().to(device)
    
    optimizer = optim.RMSprop(net.parameters(), lr=1e-3) #net.parameters are the parameters to optimise.

    #Lists in which to store hessian data.
    #These will be lists of tuples like (iteration number, time, loss, hessian data).
    manual_hessian_data = []
    library_hessian_data = []
    loss_data = []

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()                                 
        #batch_y0, batch_t, batch_y, t_list = get_batch()             
        pred_y = get_pred_y(t, net)
        loss = torch.mean(torch.abs(pred_y - true_y))        
        loss.backward(create_graph=True)                                                                     
        
        if itr % args.hessian_freq == 0 or itr==1:
          if args.library_hessian:
            print('Obtaining library hessian...')
            library_start = time.time()
            library_hessian = get_library_hessian(net)                       #get hessian with library functions   
            library_end = time.time()
            print("Time taken for library-based approach was " + str(round(library_end-library_start,2)) + "s.")
            library_hessian_data.append((itr, library_end-library_start, loss.item(), library_hessian))

          if args.manual_hessian:
            
            print('Obtaining manual hessian...')
            manual_start = time.time()

            pred_y = get_pred_y(t)
            loss = torch.mean(torch.abs(pred_y - true_y))
            grads = torch.autograd.grad(loss, net.parameters(), create_graph=True)
            parameters = optimizer.param_groups[0]['params']
          
            manual_hessian = get_manual_hessian(grads, parameters)           #get hessian with manual approach.
            manual_end = time.time()
            print("Time taken for manual approach was " + str(round(manual_end-manual_start,2)) + "s.")
            manual_hessian_data.append((itr, manual_end-manual_start, loss.item(), manual_hessian))

          else:
            pass
      

        if itr % args.test_freq == 0:
          ii += 1       
          with torch.no_grad():
              pred_y = get_pred_y(t, net)
              loss = torch.mean(torch.abs(pred_y - true_y))
              loss_data.append((itr, loss.item()))
              print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
              visualize(true_y, pred_y, net, ii)
       
        """
        else:
          if itr % 50 == 0 or itr==1:
            if args.library_hessian:
              print('Obtaining library hessian...')
              library_start = time.time()
              library_hessian = get_library_hessian(func)                       #get hessian with library functions   
              library_end = time.time()
              print("Time taken for library-based approach was " + str(round(library_end-library_start,2)) + "s.")
              library_hessian_data.append((itr, library_end-library_start, loss.item(), library_hessian))
        
          if itr % 50 == 0:
            ii += 1       
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                loss_data.append((itr, loss.item()))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii)
        """  
        optimizer.step()
        print(itr, end=',')


# In[ ]:


#torch.save(net, '/content/drive/MyDrive/colab_notebooks/calculating_hessians/testing_on_normal_nets/exponential_curve/test_3/test_3_model.pt')


# In[ ]:


itrs = []
data = []

for item in loss_data:
  itrs.append(item[0])
  data.append(item[1])

plt.figure(figsize=(10,8))
plt.rcParams.update({'font.size': 14})
plt.plot(itrs, data)
plt.xlabel('Iterations')
plt.ylabel('Loss')
#plt.savefig('/content/drive/MyDrive/colab_notebooks/calculating_hessians/testing_on_normal_nets/exponential_curve/test_3' 
            #+ '/loss_curve.png')
plt.show()


# In[ ]:


for item in library_hessian_data:
  e, v = torch.symeig(item[3])
  plt.hist(e.cpu().numpy(), bins=150)
  plt.title("Iteration: " + str(item[0]))
  plt.xlabel('Eigenvalue')
  plt.ylabel('Density')
  #plt.savefig('/content/drive/MyDrive/colab_notebooks/calculating_hessians/testing_on_normal_nets/exponential_curve/test_3' 
              #+ '/eigenvalue_density_plots/eigenvalue_density_' + str(item[0]) + '.png')
  plt.show()


# In[ ]:


itrs = []
values = []
for item in library_hessian_data:
  itrs.append(item[0])
  e, v = torch.symeig(item[3])
  value = max(e, key=abs)
  values.append(value)

plt.figure(figsize=(10,8))
plt.rcParams.update({'font.size': 14})
plt.plot(itrs, values)
plt.title('Extremal Eigenvalues for Gradient Descent')
plt.xlabel('Iterations')
plt.ylabel('Extremal Eigenvalue')
#plt.savefig('/content/drive/MyDrive/colab_notebooks/calculating_hessians/testing_on_normal_nets/exponential_curve/test_3' 
            #+ '/extremal_eigenvalues.png')
plt.show()

