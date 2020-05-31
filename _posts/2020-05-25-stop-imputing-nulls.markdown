---
layout: post
title:  "Stop Imputing Nulls?"
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

Maybe. You have a model for
$$P(y|X_1, X_2, X_3)$$
, you want a model for
$$P(y|X_1, X_2)$$
. You understandably don't want to have to train a new model for every possible combination `NULL` values. What can be said is

$$P(y|X_1, X_2) = \int dX_3 P(y|X_1, X_2, X_3)P(X_3|X_2,X_1)$$

, which by the [law of large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers), is equivalent to

$$P(y|X_1, X_2) = \mathbb{E}_{X_3\sim P(X_3|X_1,X_2)} P(y|X_1, X_2, X_3)$$

This of course raises a new question: how would one go about computing
$$P(X_3|X_1,X_2)$$ 
? This is a question of conditional density estimation, and generally speaking not an easy task (the sole exception being if all variables _except_ the ones that are null are categorical in nature. In this case, you can just condition on your categorical variables and average over null ones). However, when you impute
$$X_3$$
with
$$\hat{X}_3$$
, you're effectively assuming
$$P(X_3|X_1, X_2)=\delta(X_3-\hat{X}_3)$$
. You are also _likely_  picking a point that maximizes some reasonable conditional density estimator. For example, if you're imputing with a weighted average of nonnull values, you're probably assuming some distribution over $$X_1\times X_2$$ that peaks at that same weighted average over coordinate vectors $$(X_1, X_2)$$. Hence if you can think of a reasonable way to impute $$X_3$$, there is very likely a corresponding density estimator you would also find reasonable. Meanwhile, if
$$X_3$$
is conditionally independent of
$$X_1, X_2$$
, you can just average your model over randomly chosen samples of
$$X_3$$
regardless of its corresponding
$$X_1\times X_2$$
.

I can think of two generic yet potentially practical options.

### Binning
You could bin your nonnull continuous variables (for simplicity, let's suppose $$X_1$$ and $$X_2$$ are continuous unless stated otherwise), effectively making them categorical. Then you are free to sample values of $$X_3$$ found within a corresponding bin of $$X_1, X_2$$. This effectively approximates $$P(X_3|X_1,X_2)$$ as constant within a sufficiently narrow box in $$X_1\times X_2$$ space.

So what size bin do you want to use? Well, you're ultimately estimating a finite sample average of $$\mathrm{model}(X_1, X_2, X_3)$$ (averaged over different values of $$X_3$$), so the [Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem) dictates you're going to get a sample variance that drops off like $$\frac{1}{N}$$. Better still, your model is estimating a probability distribution, so (assumimg this is a distribution over a discrete variable $$y$$) every output of the model will be bounded between $$0$$ and $$1$$. This means you can place an upper bound of $$\frac{1}{4N}$$ on the variance of your finite sample estimate.

This means that your bin should be large enough that
$$\frac{1}{4N}$$
is an acceptable sample variance, where
$$N$$
is the number of samples found within your bin. Meanwhile, if your bin is too large,
$$P(X_3|X_1,X_2)$$
may no longer be approximately constant.

As an extreme extension, one could sample uniformly from the convex hull defined by the k nearest $$X_1, X_2$$. This, however, would require iterating over the entire dataset to identify close points, andwould likely require the same order of computation as training a new model (unless you already have the data stored in a [K-D tree](https://en.wikipedia.org/wiki/K-d_tree)).

### Kernel Regression
[Kernel density estimation](https://en.wikipedia.org/wiki/Kernel_regression) has been used as a quick way to approximate conditional expectations. The formalism looks like this:

$$\overline{z}(X) = \frac{\sum\limits_i K_h(X_i - X) z_i}{\sum\limits_i K_h(X_i - X)}$$

This is called a Nadaraya–Watson estimator, and is generally used for regression. In our case, we want to regress the expected value of $$\mathrm{model}(X_1, X_2, X_3)$$ as a function of nonnull arguments $$X_1, X_2$$.

$$\overline{\mathrm{model}}(X_1, X_2, X_3) = \frac{\sum\limits_i K_h(X_1 - X_{1,i}, X_2 - X_{2,i}) \mathrm{model}(X_1, X_2, X_{3,i})}{\sum\limits_i K_h(X_1 - X_{1,i}, X_2 - X_{2,i})}$$

We are effectively integrating over a weighted averge of existing sample values with weights that increase as you get closer to $$(X_1, X_2)$$, representing the particular values of $$X_1\times X_2$$ taken for our particular example (with null $$X_3$$). The central limit theorem again dictates $$\frac{1}{N}$$ sample variance for a finite sample estimate of $$N$$ samples.

Note $$K_h(x_1, x_2, \dotsc, x_n)=\exp\left(\frac{1}{h^2}\sum\limits_{i=1}^n x_i^2\right)$$ for a given bandwidth $$h$$. Choosing a bandwidth can be performed using [common heuristics or cross validation](https://en.wikipedia.org/wiki/Kernel_density_estimation#Bandwidth_selection), however you're back to training a new model if you use anything but a $$O(1)$$ heuristic. Of course, if you do go that route, you could make $$h$$ a vector and pick different scales for different coordinates, or even build a full covariance matrix. For example, you could build the full covariance matrix from nonnull samples, invert it, and just use this to build Nadaraya-Watson estimators when you hit null values. If some of your features are categorical, you'd have to repeat this process for every new unique combination of nonnull categories you hit, but it's still a $$O(1)$$ process if you fix the number of samples you use to build your covariance matrix ahead of time (and your data is appropriately indexed across categorical features for easy lookups).

One nice aspect about this technique is you can just randomly sample points from your existing dataset, and weight them accordingly, and expect $$\frac{1}{N}$$ sample variance guaranteed by the central limit theorem.

It's also worth pointing out that as $$h\rightarrow\infty$$ you approach assuming conditional independence between $$X_3$$ and $$X_1, X_2$$ (in which case you're just averaging the $$\mathrm{model}$$ over randomly sampled values of $$X_3$$). As $$h\rightarrow 0$$, you approach imputation with a nearest neighbor lookup.

### Multiple NULLs
When you have  more than one `NULL` feature, the correct extension of this technique would be to draw random samples from the subset of data you have for which _all_ (currently) missing values are nonnull. That is, if $$X_2$$ and $$X_3$$ were missing, you'd draw random instances in which $$X_2$$ and $$X_3$$ were _both_ nonnull, and use each of those pairs of values to simultaneously impute $$X_2$$ an $$X_3$$ for each sample. This effectively gives you

$$P(y|X_1) = \mathbb{E}_{X_2, X_3\sim P(X_2, X_3|X_1)} P(y|X_1, X_2, X_3)$$

### Regression
Regression algorithms seek to learn the expectation of the dependent variable given the independent variables.

$$\overline{y}=\int y P(y|X_1\times X_2,X_3) dy$$

The same argument can be applied as before here, now to computing $$\overline{y}$$ given $$X_1$$ and $$X_2$$ in the case of a null $$X_3$$.

## Conclusion

Rather than developing a model for missing values, you at least in principle should be trying to compute an expectation of the output of the model over a distribution governing the missing input.

As an afterthought, the above trick could be applied to training, but things would get complicated. For every `NULL` value, you'd have to bootstrap `N` new samples (a few hundred, thousand, etc) replacing the `NULL`s with randomly chosen existing values. Furthermore, you'd have to _weigh_ those bootstrapped samples with a factor of `1/N` (assuming all completely nonnull training points are assigned a weight of `1`).

This actually might not be so bad for algorithms trained via stochastic gradient descent. You might just impute the `NULL` values with a single randomly chosen nonnull example and not have to reweigh anything at all! After all, the whole point of stochastic gradient descent is to get a very rough (yet ideally unbiased) approximation of the gradient, and that's exactly what you're doing!
