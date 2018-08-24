# Gradient-Descent-Algorithms

This repository contains 3 GD algorithms
1. Basic Gradient Descent (Stochastic, Mini-Batch, Batch)
   - This is the traditional GD Algorithm.
2. Hogwild!
   - This is a lock-free parralelized implementation of Stochastic Gradient Descent (SGD). The paper for Hogwild! can be found [here](https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf)
3. Stochastic Variance Reduced Gradient Descent (SVRG)
   - This is another implementation of Stochastic Gradient Descent (SGD) where we reduce the randomness of stepping on the gradient. The Stochastic Average Gradient (SAG) Algorithm was used for this implementation. The paper for SAG can be found [here](https://arxiv.org/pdf/1309.2388v2.pdf)