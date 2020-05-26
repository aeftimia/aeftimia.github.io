---
layout: post
title:  "Stop Imputing Nulls!"
---
{% include math.html %}

## Setup
This is going to be a quick one, but I think an important note. For the purpose of this post, I'm going to assume you're evaluating a classifier on a dataset with partially missing data. That is, a handful of entries of a handful of features (at any given time) are `NULL` (i.e. the value is missing). In general, when you have a dataset with missing values, there are a lot of common practices people use to assign nonnull values to these missing values. 

To start with a more concrete example, suppose we are trying to predict $$y$$ from $${X_1, X_2, X_3}$$ and $$X_3$$ is missing. Whatever your model, it will almost certainly throw a `NaN` back at you unless every feature is nonnull. At this point there's decent choice of [common heuristics](https://scikit-learn.org/stable/modules/impute.html) you could be using to guess a reasonable value for $$X_3$$.

But let's take step a back for a second... What is our model trying to do? I mean formally, our model is trying to predict a conditional probability,

$$\mathrm{model}(X_1, X_2, X_3) \approx P(y|X_1, X_2, X_3)$$

That is, our model is our best attempt to fit a function to the conditional probability of $$y$$ given $${X_1, X_2, X_3}$$. When we have a null value, we're dealing with this:

$$\mathrm{model}(X_1, X_2, \mathrm{NULL}) \approx P(y|X_1, X_2, \mathrm{NULL})$$

When you impute a null $$X_3$$ with $$\hat{X}_3$$, you're hoping

$$P(y|X_1, X_2, \hat{X}_3) \approx P(y|X_1, X_2, X_3)$$

This is of course true if $$\hat{X}_3$$ really does approximate the missing $$X_3$$.

In general, when you impute a missing value, you're effectively assuming some distribution over 
$$P(X_3|X_1, X_2)$$
, and pulling a point estimate of $$X_3$$ from this. For example, if you use a nearest neighbor look up, you're effectively assuming
$$P(X_3|X_1, X_2)$$
is a delta function centered on the value $$X_3$$ takes on the nearest (nonnull) point in the training data as measured by its distance to $$(X_2, X_3)$$ using, say, a Euclidean metric. If you replace $$X_3$$ with its mean value, you are again assuming a delta function over $$P(X_3|X_1, X_2)$$, but this time it's just a single delta function centered at $$\overline{X}_3$$. You're not using $$X_1$$ and $$X_2$$ at all!

## Where Am I Going With This?

But is this what you really _want_? Remember, your objective is to compute a distribution over $$y$$ given $${X_1, X_2, X_3}$$. In the case of a missing $$X_3$$, __you have no data__ for $$X_3$$. So you're effectively trying to figure out what it _might_ be so you can plug that into your pretrained formula for
$$P(y|X_1, X_2, X_3)$$
. "What is the distribution of $$y$$ given $$X_1$$, $$X_2$$, and our best guess for what $$X_3$$ might have been?" is a proxy for "What what is the distribution of $$y$$ given $$X_1$$ and $$X_2$$?"--the latter being what you're actually after.

## Got a Better Idea?

Why yes, I do! You have a model for
$$P(y|X_1, X_2, X_3)$$
, you want a model for
$$P(y|X_1, X_2)$$
. You understandably don't want to have to train a new model for every possible combination `NULL` values. The good news is you don't need to because

$$P(y|X_1, X_2) = \int dX_3 P(y|X_1, X_2, X_3)P(X_3)$$

, which by the [law of large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers), is equivalent to

$$P(y|X_1, X_2) = \mathbb{E}_{X_3\sim P(X_3)} P(y|X_1, X_2, X_3)$$

That means if we just average the output of our original $$\mathrm{model}(X_1, X_2, X_3)$$ over randomly chosen nonnull values of $$X_3$$, we can answer our real question directly!

### Multiple NULLs
When you have  more than one `NULL` feature, the correct extension of this technique would be to draw random samples from the subset of data you have for which _all_ (currently) missing values are nonnull. That is, if $$X_2$$ and $$X_3$$ were missing, you'd draw random instances in which $$X_2$$ and $$X_3$$ were _both_ nonnull, and use each of those pairs of values to simultaneously impute $$X_2$$ an $$X_3$$ for each sample. This effectively gives you

$$P(y|X_1) = \mathbb{E}_{X_2, X_3\sim P(X_2, X_3)} P(y|X_1, X_2, X_3)$$

## But Isn't That Computationally Expensive?

Not necessarily! You don't need to use every sample you have of $$X_3$$, not by a long shot. In practice, you're estimating a finite sample average, so the [Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem) dictates you're going to get a sample variance that drops off like $$\frac{1}{N}$$. Better still, your model is estimating a probability distribution, so (assumimg this is a distribution over a discrete variable $$y$$) every output of the model will be bounded between $$0$$ and $$1$$. This means you can place an upper bound of $$\frac{1}{2N}$$ on the variance of your finite sample estimate.

## Conclusion

You probably shouldn't be imputing `NULL` inputs of supervised classifiers. Definitely not for inference at least. Actually the above discussion still holds for regression models, which you could argue fit
$$P(y|X)$$
for continuous $$y$$ to a delta function
$$P(y|X)=\delta\left(y-\mathrm{model}(X)\right)$$
. Things would get a little more complicated here because while the average of a sample of multinomial distributions (for categorical $$y$$) is a multinomial distribution, the average of a sample of delta functions is... the average of sample of delta functions. However, you could center a new delta function at the expectation over your mixture and effectively arrive at e.g.
$$P(y|X)=\delta\left(y-\mathbb{E}_{X_3\sim P(X_3)}\mathrm{model}(X_1, X_2, X_3)\right)$$
, which is analogous to what we derived for classifiers.

As an afterthought, the above trick could be applied to training, but things would get complicated. For every `NULL` value, you'd have to bootstrap `N` new samples (a few hundred, thousand, etc) replacing the `NULL`s with randomly chosen existing values. Furthermore, you'd have to _weigh_ those bootstrapped samples with a factor of `1/N` (assuming all completely nonnull training points are assigned a weight of `1`).

This actually might not be so bad for algorithms trained via stochastic gradient descent. You might just impute the `NULL` values with a single randomly chosen nonnull example and not have to reweigh anything at all! After all, the whole point of stochastic gradient descent is to get a very rough (yet ideally unbiased) approximation of the gradient, and that's exactly what you're doing!
