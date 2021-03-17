Test_3 description

Ground truth equation: dy/dt = exp(t), y_0 = 2, t_0 &lt; t &lt;t_1

Loss function: modulus difference, summed over all time points. In code:


```
    pred_y = odeint(func, true_y0, t)
    loss = torch.mean(torch.abs(pred_y - true_y))
```


ODESolver time steps: 1000

ODESolver method: dopri5

Neural net used: 1 hidden layer with 10 units, with tanh() activation function. 

Optimizer: <code>optim.RMSprop</code>

Learning Rate: 1e-1

Parameters Initialisation: biases = 0, weights randomly chosen using Gaussian distribution with mean 0 and std 0.1.

Number of iterations: 8000

Uses gradient descent (i.e. doesn’t use stochastic methods).

Data size = 1000

Test Frequency = 100

Hessian Frequency = 100

Hessian tool used: library-function method.
