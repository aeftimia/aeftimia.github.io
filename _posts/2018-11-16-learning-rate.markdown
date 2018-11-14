---
layout: post
title:  "Learning Rate Rates"
---
{% include math.html %}

## Introduction

$$\mathbf{\theta}_{n+1} = \mathbf{\theta}_n + \mathbf{\eta}\nabla_\mathbf{\theta} J(\mathbf{\theta}_n)$$

Ever wondered what your learning rate should be when using stochastic gradient descent? Of course you have! Everyone has! Well boy today's your lucky day! I've got the _perfect_ solution to your problem. Here's yet another adaptive learning rate algorithm. Because we don't have enough of them already.

$$\mathbf{\theta}_{n+1} = \mathbf{\theta}_n + \left(\mathbf{\eta}_{n}\otimes\nabla_{\mathbf{\theta}_{n}}\right) J(\mathbf{\theta}_n)$$

$$\mathbf{\eta}_{n+1} = \mathbf{\eta}_{n} + \left(\mathbf{\eta}^\prime\otimes\nabla_{\mathbf{\eta}_{n}}\right) J(\mathbf{\theta}_{n+1}\left(\theta_{n}, \eta_{n}\right))$$

Notice I've made $\eta$ a vector, so we're now dealing with a parameter specific learning rate. This will come in handy later when we face momentum based algorithms that track $\mathcal{O}(n)$ as many momentum parameters as free ones $n$ used in the model. Now, in the algorithm described here, we imagine at time step $n$ we simultaniously update $\theta_n\rightarrow \theta_{n+1}$, and $\eta_n\rightarrow \eta_{n+1}$ with stochastic gradient descent.

$$
\mathbf{\eta}_{n+1} = \mathbf{\eta}_{n} + \left(\mathbf{\eta}^\prime
\otimes
\nabla_{\mathbf{\eta}_{n}}\right)
J\left(\mathbf{\theta}_{n} + \left(\mathbf{\eta}_{n}\otimes\nabla_{\mathbf{\theta}_{n}}\right) J(\mathbf{\theta}_{n}) \right)
$$

$$
\mathbf{\eta}_{n+1} = \mathbf{\eta}_{n} +
\mathbf{\eta}^\prime
\otimes
\nabla_{\mathbf{\theta}_{n+1}}
J\left(\mathbf{\theta}_{n+1}\right)
\otimes
\nabla_{\mathbf{\theta}_{n}}
J\left(\mathbf{\theta}_{n}\right)
$$

Hmm... OK, so we didn't take _too_ much of a performance hit with our adaptive learning rate algorithm. Once we compute $\nabla_{\mathbf{\theta}_n} J\left(\mathbf{\theta}_n\right)$, we can just cache it and multiply it by its value from the previous time step to compute the next value of the learning rate.

Let's see what happens if we were to update the learning rate rate $\eta^{\prime}$ in a similar manner.

$$\mathbf{\eta^\prime}_{n}
= \mathbf{\eta^\prime}_{n} +
\mathbf{\eta}^{\prime\prime}
J\left(\theta_{n} + \left(
\eta_{n-1} + \eta^{\prime}_{n-1} \otimes
\nabla_{\theta_{n}}J(\theta_{n})\otimes
\nabla_{\theta_{n-1}}J(\theta_{n-1})
\right)
\otimes\nabla_{\theta_{n}} J(\theta_{n})\right)$$

$$
\mathbf{\eta^\prime}_{n}
= \mathbf{\eta^\prime}_{n-1} +
\mathbf{\eta}^{\prime\prime}
\otimes
\nabla_{\mathbf{\eta^\prime}_{n-1}}
\otimes
\nabla_{\theta_{n+1}}J(\theta_{n+1})
\otimes
\nabla_{\theta_{n}}J(\theta_{n})
\otimes
\nabla_{\theta_{n}}J(\theta_{n})
\otimes
\nabla_{\theta_{n-1}}J(\theta_{n-1})
$$

The pattern is

$$
\alpha^m_{n+1} \equiv
\frac{\mathbf{\eta}^{\left(\prime\right)^m}_{n+1} - \mathbf{\eta}^{\left(\prime\right)^m}_{n}}{\mathbf{\eta}^{\left(\prime\right)^{m+1}}_{n}}
=
\frac{\partial \mathbf{\eta}^{\left(\prime\right)^m}_{n+1}}{\partial \mathbf{\eta}^{\left(\prime\right)^{m+1}}_{n}}
=
\nabla_{\eta^{\left(\prime\right)^{m}}_{n-m}}J\left(\theta_{n+1}\right)
=
\frac{\partial J\left(\theta_{n+1}\right)}{\partial\theta_{n+1}}
\otimes
\frac{\partial\theta_{n+1}}{\partial\eta_{n}}
\otimes
\frac{\partial\eta_{n}}{\partial\eta^\prime_{n-1}}
\otimes
\cdots
\frac{\partial\eta^{\left(\prime\right)^{m-1}}_{n-m+1}}{\partial\eta^{\left(\prime\right)^{m}}_{n-m}}
$$

$$
\alpha^m_{n+1}
=
\alpha_{n}^{m-1}
\otimes
\frac{\partial\eta^{\left(\prime\right)^{m-1}}_{n-m+1}}{\partial\eta^{\left(\prime\right)^{m}}_{n-m}}
$$

$$
\alpha^m_{n+1}
=
\alpha_{n}^{m-1}
\otimes
\frac{\partial\eta^{\left(\prime\right)^{m-1}}_{n-m+1}}{\partial\eta^{\left(\prime\right)^{m}}_{n-m}}
=
\alpha_{n}^{m-1}
\otimes
\alpha_{n-m-1}^{m-1}
$$

This can be done incrementally. After each batch, update each $\alpha$ using the appropriate product from prior batches. This ultimately means the algorithm has to cache $m$ prior values of each $\alpha^m$, and therefore has $\mathcal{O}(nm^2)$ space complexity with respect to the number of parameters $n$ and order of the learning rate $m$.

The neat part is that each time we push back the learning rate, we multiply it by another gradient. Most of the time, we'll have a vanishing gradient problem, and the final hyperparameters adapt at increasingly slow rates as we increase the number of learning rates we introduce. Occasiaionlly, we'll have an exploding gradient problem, in which case you'll probably want to clip it anyway. Notice if we _didn't_ clip the exploding gradients, they'd remain cached and used to update the $n^\mathrm{th}$ learning rate for the next $n+1$ batches.

This behavior is, in theory, very analagous to what would happen in a momentum method. For the latter, we keep an exponential moving average of all the previous gradients, and depending on the time constant of that exponential, huge gradients will continue to influence the learning rates for some nonzero number of batches in the future. Also, like momentum methods, closer inspection of how many times a problem gradient enters into our formula for the next $\mathbf{\theta}$ shows that with each successive batch, a problemed gradient will influence one less learning rate [rate rate...]. So in perhaps a less explicit sense than with artificial momentum, each gradient will gradually lose influence over $\theta$ with successive batches.
