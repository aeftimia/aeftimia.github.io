---
layout: post
title:  "The Capacity of Neural Networks"
---
{% include math.html %}

## Introduction

Information theory has a rather interesting concept known as _channel capcity_, $C$. The channel capacity characterizes any _channel_ (that which produces output symbols, $y\in Y$ from input symbols $x\in X$) by the maximum mutual information you could ever hope to achieve beteween the inputs and the outputs. For instance, suppose you have a channel that takes a single input bit (i.e. one symbol that can have one of two single digit values) and produce one output bit such that the output bit is identical to the input bit with probability $f$ and is flipped the other $1-f$ time. We might write the probability that the output is 1 given the input is 1 as $p(1\vert 1)=f$, and likewise the probability of producing 0 given an input of 1 as $p(0\vert 1)=1-f$. We'll call the probability of sending a 1 $p(x=1)=s$. Note that $$p(y=1) = \sum_x p(y=1\vert x)p(x) = s f + (1-f)(1-s)$$. We can thereby characterize the mutual information between the output and input as

$$I(Y;X) = H(Y) - H(Y\vert X) = \sum_{x\in X, y\in Y}p(x) p\left(y\vert x\right) \log_2\left(p\left(y\vert x\right)\right) - \sum_{y\in Y} p(y)\log_2\left(p\left(y\right)\right) $$

$$I(Y;X) = H_2\left(sf + (1-s)(1-f)\right) - H_2(f)$$

where

$$H_2(q) = -q\log_2(q) - (1-q)\log_2(1-q)$$

is the entropy of a Bernoulli distribution characterized by probability $q$. This is maximized when $s=1/2$, giving

$$C = \underset{p}{\max} I(Y;X) = 1 - H_2(f)$$

You can perform a similar analysis on continuous variables. Suppose I encode a continuous value, $$x$$, into a channel and decode a signal at the other end given by a gaussian centered on $x$. That is

$$p(y\vert x) \sim \mathcal{N}\left(x,\sigma^{2}\right)$$

So the mutual information is

$$ I(X;Y) = \int dx dy P(x) P(y\vert x) \log_2 \left(P(y\vert x)\right) - P(y) \log_2 \left(P(y)\right) $$

$$ I(X;Y) = \int dx dy P(x) \varphi\left(\frac{y-x}{\sigma}\right) \log_2 \left(\varphi\left(\frac{y-x}{\sigma}\right)\right) - P(x) \varphi\left(\frac{y-x}{\sigma}\right) \log_2 \left(\int dx^\prime P(x^\prime) \varphi\left(\frac{y-x^\prime}{\sigma}\right)\right) $$

$$ I(X;Y) = - H_\sigma - \int dx dy P(x) \varphi\left(\frac{y-x}{\sigma}\right) \log_2 \left(\int dx^\prime P(x^\prime) \varphi\left(\frac{y-x^\prime}{\sigma}\right)\right) $$

, where $$\varphi$$ is a normal distribution centered at $0$ with unit variance and $H_\sigma = \log_2(2\pi \sigma)$ is the entropy of gaussian distribution of variance $\sigma^2$. This of course simplifies nicely if we assume $P$ has maximum entropy over the real line given fixed variance (i.e. is a normal distribution).

## Linear Activation

5 years ago, I'd jump on the opportunity to derive something beautiful. Now I pass my time through a _practicality_ filter to make sure I'm not just mentally jacking myself off. This exercise actually turns out to be useful, so today I get to jack off...

So what happens when our channel is a simple linear transformation? $P(y\vert x) = \delta\left(y-\textbf{A} x\right)$. That is, we're looking at a dirac delta distribution centered around a linear transformation of the input, $x$. That's not terribly interesting... Or is it!? [cue dramatic music]

We can backtrack to calculate
$$P(y) = \int dx P(y\vert x)P(x)=\sqrt{\lvert\mathbf{A}^\top \mathbf{A}\rvert}\int_{x^\prime\in \mathrm{Ker}\left(\mathbf{A}\right)}P\left(X=x^\prime + \textbf{A}^\dagger y\right)$$,

Since the pseudoinverse won't give us anything in kernel space, we have to integrate over the kernel space to account for possible nullspace components within the preimage. In principal, this could be part of a nontrivial joint distribution of the components of $x$ that unequally favors null components given different values along other components. The determinent in front just rescales the proability density (remember, we *still* end up with a probability density over $y$ after integrating over $x^\prime$) $P$ over $X$

Since the channel is completely deterministic, the first term that composes the mutual information,

$$\sum_{x\in X, y\in Y}p(x) p\left(y\vert x\right) \log_2\left(p\left(y\vert x\right)\right) = 0 $$

Why? Because $p$ is going to become increasingly narrow given full knowledge of the inputs and no noise in between the inputs and outputs. This is going to shrink the entropy to zero. So on to the next term!


$$I(Y;X) = -\sqrt{\lvert\mathbf{A}^\top \mathbf{A}\rvert}\int_{x^\prime\in \mathrm{Ker}\left(\mathbf{A}\right)}P\left(X=x^\prime +\textbf{A}^\dagger y\right) \log_2\left( \sqrt{\lvert\mathbf{A}^\top \mathbf{A}\rvert}\int_{x^\prime\in \mathrm{Ker}\left(\mathbf{A}\right)}P\left(X=x^\prime + \textbf{A}^\dagger y\right) \right)$$

In general, the information in output $Y$ is a subset of the information initially available in $X$. Why? Since there's no noise to add new information, the most interesting thing we could do is compress the input with loss by mapping it to a lower dimensional space than it started. Hence the mutual information is just the remaining entropy of $Y$. If you really want to be an asshole, you can use Jenson's inequality to relate the sum of probabilities under the log to a sum of logs of those probabilities and cauchy schwartz to relate the result back to the entropy of $X$. But I'm going to try my best not to be an asshole.

If $\mathbf{A}$ is invertible, the integral can be discarded and we're left with 

$$I(Y;X) = -\sqrt{\lvert\mathbf{A}^\top \mathbf{A}\rvert}P\left(X=\textbf{A}^{-1} y\right) \log_2\left(\sqrt{\lvert\mathbf{A}^\top \mathbf{A}\rvert}P\left(X=\textbf{A}^{-1} y\right)\right)$$

## Nonlinear Transformations

OK, I dragged you through all that and you're still not sure how this isn't just me wanking to math. Well, it turns out we can generalize nicely to nonlinear functions from the linear framework we just constructed, but we have to assume monotonicity. This still isn't _totally_ useless since lots of common and useful nonlinear activition functions are monotonic (like sigmoids and the ReLUs that don't take absolute values). You'll notice the last experssion inverts $y$ back to $x$ under the probability density function, and rescales the resulting density function with the inverse of the derivative of the map $\mathbf{A}^{-1}:y\rightarrow x$ so it integrates right over the new variable. We can do that for nonlinear functions too if we assume monotonicity. Ok, we also have to pretend critical points don't exist, but as a matter of practice they really don't. Asshole mathematicians used to look at deep nets and scream bloody murder over all the local minima to get trapped in. It turns out hitting measure zero points in high dimensional spaces is... well, you do the math.

$$I(Y;X) = -\sqrt{\lvert\mathbf{A}^\top \mathbf{A}\rvert}P\left(X=\textbf{A}^{-1} y\right) \log_2\left(\sqrt{\lvert\mathbf{A}^\top \mathbf{A}\rvert}P\left(X=\textbf{A}^{-1} y\right)\right)$$
