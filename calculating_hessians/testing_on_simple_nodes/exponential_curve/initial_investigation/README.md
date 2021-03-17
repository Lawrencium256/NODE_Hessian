Initial Exponential Curve NODE Description

Ground truth equation: dy/dt = exp(t), y_0 = 2, t_0 &lt; t &lt;t_1

Loss function: modulus difference, summed over all time points. In code:


```
    pred_y = odeint(func, true_y0, t)
    loss = torch.mean(torch.abs(pred_y - true_y))
```


ODESolver time steps: 1000

ODESolver method: dopri5

Neural net used: 1 hidden unit, with tanh() activation function. 

Optimizer: <code>optim.RMSprop</code>

Learning Rate: 1e-2

Parameters Initialisation: biases = 0, weights randomly chosen using Gaussian distribution with mean 0 and std 0.1.

Number of iterations: 15000

Batch size = 20

Batch time = 10

Data size = 1000

Hessian tool used: library-function method.
