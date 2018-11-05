---
layout: post
title:  "On the Channel Capacity of Deep Neural Networks"
---
{% include math.html %}

## Introduction

Electrical engineers have a rather interesting information theoretic concept known as "channel capcity", $C$. The channel capacity characterizes any "channel" (that which produces output symbols, $y\in Y$ from input symbols $x\in X$) by the maximum mutual information you could ever hope to achieve beteween the inputs and the outputs. For instance, suppose you have a channel that takes a single input bit (i.e. one symbol that can have one of two single digit values) and produce one output bit such that the output bit is identical to the input bit with probability $f$ and is flipped the other $1-f$ time. We might write the probability that the output is 1 given the input is 1 as $p(1\vert 1)=f$, and likewise the probability of producing 0 given an input of 1 as $p(0\vert 1)=1-f$. We'll call the probability of sending a 1 $p(x=1)=s$. Note that $$p(y=1) = \sum_x p(x,y) = s f + (1-f)(1-s)$$. We can thereby characterize the mutual information between the output and input as

$$I(Y;X) = H(Y) - H(Y|X) = \sum_{y\in Y} - p(y)\left(\log2(p\left(y\right)\right) - \sum_{x\in X, y\in Y}p(x) p\left(y|x\right)log2\left(p\left(y|x\right)\right) = \left(sf + (1-f)(1 - s)\right)\log2 \left(sf + (1-f)(1 - s)\right) +  \left(1 - \left(sf + (1-f)(1 - s)\right)\right)\log2\left(1 - \left(sf + (1-f)(1 - s)\right)\right)\right) - \sum_{x\in X}-s(x)\left(f\log2(f) + (1-f)\log2(1-f)\right) = H2\left(sf + (1-s)(1-f)\right) - H2(f)$$

where $$H2(q) = -q\log2(q) -(1-q)\log2(1-q)$$ is the entropy of a Bernoulli distribution characterized by probability $q$. This is maximized when $s=1/2$, giving

$$C = maxp_{p} I(Y;X) = 1 - H2(f)$$

You can perform a similar analysis on continuous variables. Suppose I encode a continuous value, $$x$$ into a channel and decode a signal at the other end given by a guassian centered on $x$. That is

$$p(y|x) = \mathcal{N}(x,\sigma^{2})$$

The entropy of a guassian ends up being (I'll save you the details) $\log2\left(2\pi e\sigma)$. So the mutual information is $I(X;Y) = H(X) - \log2\left(2\pi e\sigma)$. If we fix the variance of $X$, we end up getting a maximum entropy distribution of $X$ over its domain 

