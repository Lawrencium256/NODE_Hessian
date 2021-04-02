## Nested Spheres with Cross Entropy Loss

The key things seen from these experiments are as below:

</int>Experiment 0</int>

* ANODE top eigenvalue grows more quickly than that of the NODE. It then decays to a smaller value later in training, and does so more smoothly
* NODE possesses a 'spike' which is associated with a jump in the loss surface.
* 2nd eigenvalue for ANODE also decays more quickly than for the NODE. Again, it does so more smoothly.
* The curves of minimum eigenvalue are roughly comparable, although that of the ANODE is smoother and perhaps slightly smaller.

<u>Experiment 1</u>

* Same thing with the ANODE top eigenvalue growing quicker, and then decaying.
* Didn't see any isolated spikes of top eigenvalue.
* Same thing with the 2nd eigenvalue.
* Minimum eigenvalue for ANODE decays towards 0 more quickly a smoothly than for the NODE.



<u>Experiment 2</u>

* ANODE top eigenvalue decays more towards the end of training. But the NODE grows quicker in the initial stages, unlike in experiments 0 & 1.
* Don't see any isolated spikes in top eigenvalue.
* Same thing for the 2nd eigenvalues.
* Minimum eigenvalue does the same thing as in experiment 1.

<u>Experiment 3</u>

* Same thing with top eigenvalue decay, although in this case the ANODE top eigenvalue decays to a value significantly smaller than in previous repeats.
* Same thing with the 2nd and minimum eigenvalues.
