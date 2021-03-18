## Hessian Calculation Experiments

These files are used to calculate the Hessian during training of a NODE. There are 3 different approaches that can be used:

1. Finite Differences Approach:  a crude numerical approximation to the second derivative. This is explained in more detail on the Wikipedia page (https://en.wikipedia.org/wiki/Finite_difference). It is the slowest approach and is often sensitive to the pertubation parameter, but doesn't require backpropagating through the ODE solver.
2. "Library" approach: the fastest method, and more reliable than finite differences. This backpropagates twice through the ODE solver, and does so using PyTorch functionality via `torch.autograd.functional.hessian()`.
3. "Manual" approach: the same basic idea as the "library" approach, except the double backpropagation is coded explicity using `torch.autograd.grad()`.

The code used to develop these approaches can be found in the folders **mofd_hessian**, **library_function_hessian** and **manual_hessian**. The folder **testing_on_normal_nets** contains code to compare the Hessian calculation in the context of ordinary neural networks (NNs) with an analytical expression that can be easily obtained. It also contains an experiment on fitting an exponential curve with an ordinary NN. The folder **testing_on_simple_nodes** contains multiple experiments to investigate the Hessian during training on small-scale NODEs.
