## Results Description

<u>Experiment 0</u>

* Hessian calculations were only every 20 iterations (vs every 5 for the remaining experiments).
* Both the NODE and ANODE learned well, reaching exactly 0 loss.
* The top eigenvalue increases and then drops to 0, since the hessian reaches a matrix of zeros once the loss is 0.
* ANODE top eigenvalue is generally larger than the NODE top eigenvalue.
* The minimum eigenvalues tend to 0, although the ANODE is much more negative earlier on.
* The trace seems to be dominated by the top eigenvalue.

<u>Experiment 1</u>

* NODE learns well; ANODE learns poorly.
* ANODE top eigenvalue never really grows. The spectrum looks like it did upon initialisation. NODE behaviour is similar to experiment 0.
* The maximum trace ratio is slightly greater than 1 for the NODE. It drops to 0/0 later on.

<u>Experiment 2</u>

* ANODE learns well; NODE learns poorly. Although the ANODE does have quite a lot of fluctuation immediately before reaching 0.
* ANODE behaves as in experiment 0. The NODE acts strangely, with a large positive outlier.
* For on point in training, the NODE has a massive negative eigenvalue spike.

<u>Experiment 3</u>

* ANODE learns well; NODE learns well but doesn't reach 0. Instead, the loss tends to smaller and smaller values.
* ANODE top eigenvalue is as expected. NODE flucutates around ~100 in a strange fashion.
* Trace ratio peaks at a value greater than 1 (~1.75).

<u>Cross Entropy, Experiment 0</u>

* Both NODE and ANODE train well, but don't reach 0 as when using Hardtanh loss.
* Loss curve occassionally fluctuates in a strange way for NODE. This coincides with a big spike in the top eigenvalue in the NODE. This is not seen at other points in the spectrum, however, even when there is a corresponding loss jump.
* The ANODE top eigenvalue generally grows more quickly than the NODE. Later on in training, it seems to decay such that it as a smaller value than the ANODE.
* 2nd maximum eigenvalue for ANODE decays to 0. This is not the same for the NODE.
* Minimum eigenvalue for both decays to 0.
