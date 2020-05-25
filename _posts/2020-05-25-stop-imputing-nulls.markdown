---
layout: post
title:  "Stop Imputing Nulls!"
---
{% include math.html %}

# Setup
This is going to be a quick one, but I think an important note. For the purpose of this post, I'm going to assume you're training a classifier on a dataset with partially missing data. That is, a handful of entries of a handful of features (at any given time) are `NULL` (i.e. the value is missing). In general, when you have a dataset with missing values, there are a lot of common practices people use to assign nonnull values to these missing values. 

To start with a more concrete example, suppose we are trying to predict $$y$$ from $${X_1, X_2, X_3}$$ and $$X_3$$ is missing. Whatever your model, it will almost certainly throw a `NaN` back at you unless every feature is nonnull. At this point there's decent choice of [common hueristics](https://scikit-learn.org/stable/modules/impute.html) you could be using to guess a reasonable value for $$X_3$$.

But let's take a back for a second... What is our model trying to do? I mean formally, our model is trying to predict a conditional probability,

$$\mathrm{model}(X_1, X_2, X_3) \approx P(y|X_1, X_2, X_3)$$

That is, our model is our best attempt to fit a function to the conditional probability $$y$$ given $${X_1, X_2, X_3}$$. When we have a null value, we're dealing with this:

$$\mathrm{model}(X_1, X_2, \mathrm{NULL}) \approx P(y|X_1, X_2, \mathrm{NULL})$$

When you impute a null $$X_3$$ with an imputed $$\hat{X}_3$$, you're hoping

$$P(y|X_1, X_2, \hat{X}_3) \approx P(y|X_1, X_2, \mathrm{NULL})$$

This is of course true if $$\hat{X}_3$$ really does approximate the missing value.

In general, when you impute a missing value, you're effectively assuming some distribution over $$P(X_3|X_1, X_2)$$, and pulling some maximum likelihood estimate of $$X_3$$ from this. For example. if you use a nearest neighbor look up, you're effectively assuming $$P(X_3|X_1, X_2)$$ is a delta function centered on the value $$X_3$$ takes on the nearest (nonnull) point in the training data as mesured by its distance to $$(X_2, X_3)$$ using, say, a Euclidian metric. If you replace $$X_3$$ with its mean value, you are again assuming a delta function over $$P(X_3|X_1, X_2)$$, but this time it's just a single delta function centered at $$\overline{X}_3$$. You're not using $$X_3$$ at all!

## Where Am I Going With This?

But is this what you really _want_? Remember, your objective is to compute a distribution on $$y$$ given $${X_1, X_2, X_3}$$. In the case of a missing $$X_3$$, __you have no data__ $$X_3$$. So you're effectively trying to figure out what it _might_ be to plug it into your pretrained formula for $$P(y|X_1, X_2, \hat{X}_3)$$, where your imputed value ,

$$\hat{X}_3 = \mathrm{argmax}_{X_3} P(X_3|X_1, X_2)$$ 

I think there's a strong argument that answering "What is the distribution of $$y$$ given $$X_1$$, $$X_2$$, and our best guess for what $$X_3$$ might have been?" isn't quite what you ultimately _want_ to answer in the face of `NULL` values. I think it's a question that will, to a decent approximation, answer your _real_ question, but your real question is more like "What what is the distribution of $$y$$ given $$X_1$$ and $$X_2$$". Since $$X_3$$ is `NULL`, we simply don't have any data for it. Therefore, we try to get our best estimate of $$P(y)$$ with our best guess of what it might have been.

## Got a Better Idea?

Why yes, I do! You have a model for $$P(y|X_1, X_2, X_3)$$, you want a model for $$P(y|X_1, X_2)$$. You understandably don't want to have to train a new model every time you get a missing null. The good news is you don't need to because

$$P(y|X_1, X_2) = \int dX_3 P(y|X_1, X_2, X_3)P(X_3)$$

By the law of large numbers, this is equivalent to

$$P(y|X_1, X_2) = \mathbb{E}_{x\sim P(X_3)} P(y|X_1, X_2, X_3)$$

That means if we just average the output of our original $$model(X_1, X_2, X_3)$$ over randomly chosen nonnull values of $$X_3$$, we can answer our _real_ question _directly_.

### Multiple NULLs
When you have  more than one null missing, the correct extension of this technique would be to draw random samples in which _all_ (currently) missing values are nonnull. That is, if $$X_2$$ and $$X_3$$ were missing, you'd draw random instances in which $$X_2$$ and $$X_3$$ were _both_ nonnull, and use each of those pairs of values to simultaneously impute $$X_2$$ an $$X_3$$ for each sample. This effectivley gives you

$$P(y|X_1) = \mathbb{E}_{x\sim P(X_2, X_3)} P(y|X_1, X_2, X_3)$$

## But Isn't That Computationally Expensive?

Not necessarily! You don't need to use every sample you have of $$X_3$$, not by a long shot. In practice, you're estimating a finite sample average, so the Central Limit Theorem dictates you're going to get a sample variance that drops off like $$1/N$$. Better still, your model estimating a probability distribution. That means (assumimg this isn't a distribution over a continuous variable $$y$$), every output of the model will be bounded between $$0$$ and $$1$$. This means you can place a hard upper bound on the variance of your finite sample estimate with $$\frac{1}{2N}$$.

## Conclusion

You probably shouldn't be imputing null values for inputs of supervised models. Definately not for inference at least.

As an afterthought, the above trick could be applied to training, but things would get complicated. For every `NULL` value, you'd have to bootstrap `N` new samples (a few hundred, thousand, etc) replacing the `NULL`s with randomly chosen existing values. Furthuremore, you'd have to _weigh_ those bootstrapped samples with a factor of `1/N` (assuming all completely nonnull training points are assigned a weight of `1`).

This actually might not be so bad for algorithms trained via stochastic gradient descent. You might just impute the null values with a single randomly chosen nonnull example and not have to reweigh anything at all! Afterall, the whole point of stochastic gradient descent is to get a very rough (yet ideally unbiased) approximation to the gradient, and that's exactly what you're doing!
