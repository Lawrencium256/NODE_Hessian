### README: Sinusoidal Curve Fitting

These are the results associated with Hessian calculations for a NODE designed to fit 2 coupled sinusoidal 1D ODEs. 

The NODE is trained using the file **sinusoidal_curve.ipynb**. The architecture used in both cases is that of a 2D input and output space, with a 5D latent space.

Other details are as follows:

Solver: **<code>dopri5</code></strong>

Batch Size, Time: Varied

Number of Iterations: Varied

Test Frequency: 100

Data Size: 1000

Activation Function: Tanh()

Learning Rate: 1e-3

There are 2 similar but different experiments that have been run:

<span style="text-decoration:underline;">1000 iters:</span>


    As above, except the batch size and time are 40 and 20 respectively. The system is trained for 1000 iterations.

<span style="text-decoration:underline;">8000 iters:</span>


    As above, except the batch size and time are 200 and 50 respectively. The system is trained for 8000 iterations.

The Hessians for the models saved (at the test frequency) were calculated using the 3 available approaches: the MOFD, library-based and “manual” approaches. This is done in a separate file - **hessians_sinusoidal.ipynb**.

In all cases except 1 (for iteration 7700 in the _8000 iters_ experiment), these were found to agree, provided a reasonable choice of _h _was made for the MOFD. The evidence for this is shown in the **eigenvalue_densiy_plots_comparisons** folder. 

In addition to this, further checks were made for a larger number of models by comparing the _first 5_ elements of the calculated Hessians. This allowed a larger number of models to be validated. This was done for the _8000 iters_ experiment for the following iterations:

1000, 1200, 1300, 2000, 2100, 2500, 2800, 3000, 3100, 3200, 3600, 6100, 6400, 6600, 6900, 7000, 7400, 7800.

For iteration 7700, it was found that the MOFD doesn’t produce a Hessian similar to that obtained from other methods for any value of _h_. This is shown in the relevant histograms, and the figure  **7700_first_5_differences.png**.

In addition to the above, a slightly more general system was developed, such that it possesses a different frequency and amplitude. A NODE was trained to learn this ODE using an architecture with a 7D latent space. This corresponds to the folder **more_developed_system**.

Within this, there is the <span style="text-decoration:underline;">12000 iters</span> experiment. This corresponds to the first attempt to train the NODE. The system hyperparameters used were the same as in the <span style="text-decoration:underline;">8000 iters </span> and <span style="text-decoration:underline;">1000 iters</span> experiments. It was found that the system did not learn well (see 13/03/21 in journal).

An investigation into the effect of learning rate was then performed. The loss curves obtained are in the folder **lr_investigation**. There are 5 graphs:



*   loss_curve_1.png - learning rate: 1e-2
*   loss_curve_2.png - learning rate: 1e-3
*   loss_curve_3.png - learning rate: 5e-4
*   loss_curve_4.png - learning rate: 1e-4
*   loss_curve_5.png - learning rate: 3e-4

From these, it can be seen that a learning rate of 3e-4 is the most successful.

This more developed system was trained using a learning rate of 3e-4 until iteration 4400 (by which time it had reached an acceptable loss). It was then fine tuned using exact gradient descent, with a learning rate of 1e-6, for 200 further iterations. The results of this are contained in the folder **more_developed_system/experiments**.

Hessian analysis for this more complex case takes considerably longer. This can be done in a separate file, called **hessians_developed_sinusoidal.ipynb**, which is designed to help ease these constraints.
