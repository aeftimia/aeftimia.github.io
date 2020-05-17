---
layout: post
title:  "Stuff I've Learned in Data Science"
---
{% include math.html %}

## Introduction

I set out to study machine learning and transition into data science about three years ago now after a formal education in physics and a job writing computer simulations. In short, I came upon [this paper](https://arxiv.org/abs/1702.05532) and a few others and felt strongly that, in the mid term, machine learning would become a superset of anything practical about computational physics. I studied a lot of papers and read books on machine learning and information theory. I also studied some aspects of computer science that seemed useful in industry. While I picked up a few sorting algorithms, I actually found parsing text to be generally more useful, as this can be applied to refactoring legacy code. I learned to use vim with tmux, and got good with git. I automated a lot of my more repetitive work as a data engineer and gave myself more time to study machine learning research. I learned a bit about image processing, semisupervised learning, as well as some interesting details about how decision trees effectively make use of submodular optimization. Fast forward another two years and I've been a "data scientist proper" (that is, my primary responsibility involves creating and inspecting algorithms, applying statistics, etc) for about a year now. I like it a lot, and I feel like this has been a wonderful career change. I spend a large amount of time reading papers, some time implementing or adapting what I read, some time deriving convergence rates, and make good use of the software engineering and computer science fundamentals I picked up as a data engineer. I also get to program in Python, which I have always felt most comfortable with.

I think one of the hardest parts about starting a career or entering into a new one is getting a sense of important questions within the field. I think this is part of what is sought through formal education, but I maintain the overlap is far from perfect. Once you really get a sense of what people want you to do you can dedicate more directed effort studying for those kinds of questions. At least within my current job, I've picked up a lot more statistics than I initially would have imagined, and found formal proofs surprisingly useful. While they are still fresh in my mind, this post is going to be about all the odd tips and tricks I've picked up over the last year--much of which do not seem covered well in existing blogs.

## Never Use Table Aliases
My first lesson of data science, back from mid 2018, was strictly concerning SQL practices. My company had tens of thousands of lines of legacy SQL--a few dozen queries with around thousand lines each. They had moved and renamed dozens of columns and tables, and they wanted their old SQL updated accordingly. It was pretty miserable work. Not even miserable because it was slow and tedious. It was miserable because it could have been done in about a day had the author of the SQL code never used table aliases.

Consider the following query.

```
select 
employeeid,
a.cust,
b.date,
c.first_name,
c.last_name
on employee as a
inner join customers as b
using(transactionid)
left join transactions c
on c.transactionid = b.transactionid
```

Don't think too hard about it. I made it up and what it does isn't important. Just imagine you were told `transactions.first_name` and `transactions.last_name` have moved to `transactions_table.firstname` and `transactions_table.lastname` respectively. Meanwhile, `transactions_table` is indexed on `transaction_datetime_id` instead of `transactionid`. Sure, you could spend a couple of minutes backtracking this new query and swapping out `transactions` for `transactions_table`, the join involving `c.transactionid` for `c.transaction_datetime_id`, and finally `firstname` and `lastname` for `first_name` and `last_name` respectively.

Now consider the following query.

```
select 
employee.employeeid,
employee.cust,
customers.date,
transactions.first_name,
transactions.last_name
on employee
inner join customers
on employee.transactionid = customers.transactionid
left join transactions
on transactions.transactionid = customers.transactionid
```

Now imagine making the same changes. It should be immediately clear you can just run a few blind substitutions on the query.

* `transactions.transactionsid` => `transactions_table.transaction_datetime_id`
* `transactions.first_name` => `transactions_table.firstname`
* `transactions.last_name` => `transactions_table.lastname`
* `(?<![A-Za-z0-9.])transactions(?![A-Za-z0-9])` => `transactions_table`

And that's it. This may not seem like a big difference for a dozen lines, but when you're dealing with dozens of tables, hundreds of columns, and thousands of lines, it's the difference between a day's worth of work and *years*. I'm not exaggerating in the slightest.

I ended up writing a SQL parser to de-alias tables and query the schemas when columns had no alias at all to try to deduce which table it may have referred to. I can't say it was particularly enjoyable experience, nor can I say I did it very well, but I can say it taught me a lot--both about parsing code and about scalable practices when working with SQL.

## More Than You Ever Wanted to Know about ROC (+AUC)

You probably heard of [ROC AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) at some point regardless of whether you use it regularly. The receiver operating characteristic describes the relationship between true positive rate (proportion of positive instances labeled as positive) and false positive rate (proportion of negative instances labeled as positive). The curve itself shows this relationship as a function of threshold chosen with the left being a threshold of `1` and the right being a threshold of `0`. It takes a bit of meditating to appreciate the significance of this. If you sit on a yoga mat and actually think about it for a while, you'll see that you could construct such a plot by sorting your dataset by model score--highest to lowest--and going down the list one element at a time. You draw the ROC starting from the origin, and incrementing your curve _up_ when you find a positive instance and _to the right_ when you find a negative instance. This is in fact how [ROC AUC is computed under the hood](https://www.researchgate.net/publication/222511520_Introduction_to_ROC_analysis) (with some additional logic for tie breakers).

Meanwhile, the area under this curve is equivalent to the probability of a randomly chosen positive instance receiving a higher model score than a randomly chosen negative instance. This is an incredibly useful ranking metric! It's entirely independent of thresholds and (despite common misconceptions) class imbalances. Unfortunately, it's not differentiable, so it's not obviously applicable as a loss function for algorithms trained via gradient descent. However it might potentially work as an objective for training trees based algorithms, which are generally trained on [submodular monotone](https://en.wikipedia.org/wiki/Submodular_set_function#Monotone) objectives like information gain and [Gini impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity) (so they can be trained greedily on and attain $$1 - 1 / e$$ its globally optimum loss for a fixed depth). Not sure if ROC AUC is submodular and monotone but it might work nonetheless.

If we start out by defining ROC AUC as the probability that a randomly chosen positive instance will receive a higher model score than randomly chosen negative one, we can develop an alternative means of computing ROC AUC. We could imagine bootstrapping random pairs of positive and negative instances, scoring them, and computing the proportion of randomly selected pairs with a higher model score given to the positive instance. In practice we might iterate over all unique combinations for a $$O(N)$$ approach (where $$P$$ is the total number of positive instances and $$N$$ is the total number of negative instances). As it turns out, this is entirely equivalent to the more efficient method of sorting the instances by model score, building the ROC curve and computing the integral under it. I found it helpful to picture the bootstrapping method as the [Lebesgue](https://en.wikipedia.org/wiki/Lebesgue_integration) version of this integral--with each horizontal rectangle corresponding to a comparison of a positive model score against all negative ones (the width of the rectangle being the proportion for which the positive score was higher than the negative ones). Anywho, this exact equivalence implies by the [central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem) that ROC AUC converges like $$1/\sqrt{PN}$$, which would make it generally converge at least as fast as any one quadrant of the confusion matrix (which in turn converge like $$1/\sqrt{P}$$ or $$1/\sqrt{N}$$. This knowledge proved handy in assessing how much labeled data was needed to determine model performance for situations at work in which labels cost a great deal of manual effort.

The other neat consequence of this interpretation is it implies a $$N\log(N)+M\log(M)$$ means of computing the probability of a randomly chosen number from a list of size $$N$$ being larger than a randomly chosen element from a list of size $$M$$. If the lists are already sorted, you basically just have to go through the motion of merging them, and can compute this quantity in linear time. This effectively means a lot of questions you might have been taught to answer with a hypothesis test comparing averages can be more directly answered in the same or close to the same time complexity.

For example, you're given two sorted list of salaries. You want to know if one demographic is paid higher than another demographic. People often compare means and use a hypothesis test to determine how confident they can be that one mean is higher than the other mean. But this isn't what we want! A couple of outliers can skew averages, but that doesn't mean you need to subjectively try to filter out these people either. You can determine directly how likely it is that someone from one class would make more than someone from another class using a procedure analogous to computing ROC AUC (sort salaries within each class and merge).

## Random Forest Feature Importance

A lot of data scientists know random forests can be used to extract a measure of feature importance, but how this works seems to be less common knowledge. Since each node in a decision tree determines a split that reduces the entropy (/impurity/etc) of the target variable conditioned on all the splits leading up to and including that node. By calculating the proportion of the entropy drop attributable to splits on each feature, you can naturally derive a notion of feature importance from tree based models.

Permutation based feature importance and Sobol sensitivity analysis are other such techniques.

## Statistical Distances

[Statistical distances](https://en.wikipedia.org/wiki/Statistical_distance) how different two probability distributions are. You've probably heard of [KL-divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence). Well there is a more general notion of this kind of thing. In general, [$$f$$-divergences](https://en.wikipedia.org/wiki/F-divergence) encompass KL-divergence and anything that is invariant with respect to coordinate transformations. Not all are proper metrics. Meanwhile, [integral probability metrics](https://arxiv.org/pdf/0901.2698.pdf) encompass a set of proper metrics that can be written in the form

$$\sup_{f\in\mathcal{F}}  \left\vert\mathbb{E}_{x \sim p}\left[f(x)\right] - \mathbb{E}_{x \sim q}\left[f(x)\right]\right\vert$$

, where $$p$$ and $$q$$ are the distributions being compared, and $$\mathcal{F}$$ is some sufficiently general class of functions. Generally, bounded continuous functions are sufficient to ensure the statistical distance is $$0$$ iff $$p=q$$, but there are useful metrics that use more restricted classes of funcions. For example, if $$\mathcal{F}$$ includes only functions of [Lipschitz constant](https://arxiv.org/pdf/0901.2698.pdf) $$1$$, then you end up with the [Wasserstein metric](https://en.wikipedia.org/wiki/Wasserstein_metric) aka [Earth Mover's Distance](https://en.wikipedia.org/wiki/Earth_mover%27s_distance). This the result of an [interesting connection with linear programming](https://vincentherrmann.github.io/blog/wasserstein/).

All integral probability metrics are proper metrics, but there are statistical distances that are proper metrics but cannot be expressed as an integral probability metric (like Hellinger Distance). [Total Variation](https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures) is the only nontrivial metric that is both an $$f$$-divergence and an integral probability metric. It also has a nice interpretation as one half the hypervolume between the two pdfs being compared (and is therefore bounded between 0 and 1).

There is a very interesting connections between $$f$$-divergences and convex analysis. In general, an $$f$$-divergence can be expressed as a supremum over functions that maximize an expected loss function.

We first define the [convex conjugate](https://en.wikipedia.org/wiki/Convex_conjugate) as

$$f^{*}(y) = \sup\limits_g\left[gy - f(g)\right]$$

The [Fenchel–Moreau theorem](https://en.wikipedia.org/wiki/Fenchel%E2%80%93Moreau_theorem) states that if $$f$$ is a [proper convex function](https://en.wikipedia.org/wiki/Proper_convex_function) and [lower semicontinuous](https://en.wikipedia.org/wiki/Semi-continuity), then $$f^{**}(g) = f(g)$$.

If we let $$y=\frac{p(x)}{q(x)}$$ for some point, $$x$$, within the sample space of $$p$$ (which is assumed to be the same as that of $$q$$),

$$f\left(\frac{p(x)}{q(x)}\right) = \sup\limits_g\left[g\frac{p(x)}{q(x)} - f^{*}(g)\right] \ge g\frac{p(x)}{q(x)} - f^{*}(g)$$

, where the inequality follows from the definition of the [supremum](https://en.wikipedia.org/wiki/Infimum_and_supremum). Furthermore, if we tether $$g$$ to that same sample, $$x$$, in the form of $$g(x)$$, we get 

$$f\left(\frac{p(x)}{q(x)}\right) \ge g(x)\frac{p(x)}{q(x)} - f^{*}(g(x))$$

Multiplying both sides by $$q(x)$$

$$f\left(\frac{p(x)}{q(x)}\right)q(x) \ge g(x)p(x) - f^{*}(g(x))q(x)$$

Integrating both sides over $$x$$ drawn from the sample space, $$\mathcal{X}$$,

$$\int\limits_{x\in\mathcal{X}} dx f\left(\frac{p(x)}{q(x)}\right)q(x) \ge \int\limits_{x\in\mathcal{X}} dx g(x)p(x) - f^{*}(g(x))q(x)$$

Finally, we can apply the [law of large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers)to turn integrals over probability density functions into expectations.

$$\mathbb{E}_{x \sim q} f\left(\frac{p(x)}{q(x)}\right) \ge \mathbb{E}_{x \sim p} g(x) - \mathbb{E}_{x \sim q} f^{*}(g(x))$$

This final result shows that an $$f$$-divergence can be expressed as a function optimization problem. In this case, the right hand side of the inequality prescribes a loss function with which we use to find a function $$g(x)$$. Specifically, the loss function used would be the negative of the right hand side. When the loss is minimized, the right hand side is maximized and the inequality approaches equality to the left hand side (i.e. the true $$f$$-divergence).

You can then just chuck a neural net at this loss function and optimize. This neural network implicitly learns the density ratio of $$p$$ and $$q$$, bypassing the need to approximate each density individually. This is one of a whole class of techniques for learning density ratio approximation. There's a good [book](https://www.amazon.com/Density-Ratio-Estimation-Machine-Learning/dp/0521190177u) on the subject. This technique is also nice because it yields unbiased gradients.

Anywho, I ended up writing a library that implements this technique with some adaptations for mixed floating point and categorical data (as well as purely categorical data via histograms). I also implemented versions using KNN density ratio estimators and added some bias correction terms along the lines of the ones used in the [KSG](http://proceedings.mlr.press/v38/gao15.pdf) estimator. This library was successfully used to identify a day in which a data processing pipeline at FINRA had a bug b y comparing total variation of samples from that pipeline accross pairs of days throughout a particular month. Most samples were within 0.01 of each other as measured by total variation. One particular day was around 0.4 with respect to all the other days. After we investigated further, we found it was due to a bug in the pipeline.

## Lipschitz Constants, GANs, and Robustness
The Lipschitz constant of a continuous function is its maximum rate of change in any direction. This notion generalizes to arbitrary metric spaces in the domain and codomain. This notion is actually quite useful for applications of deep learning. As I mentioned in the previous section, bounding the Lipschitz constant of a neural network and training the resulting network on a certain loss function will learn a well defined metric between probability distributions known as the Wasserstein metric. This can be used as a critic for training a [GAN and has some very useful properties](https://arxiv.org/pdf/1704.00028.pdf). However, there is another, and I think even more useful application.

We can define the notion of adversarial robustness as the maximum difference in output for a specified change in input (typically with respect to $$L^2$$ or $$L^\infty$$ norms). This encodes the notion that the effect of small perturbations on the output of the network is fundamentally limited by this upper bound. For continuous functions--like those produced by neural networks--this is exactly the same as the Lipschitz constant, so if we can bound the Lipschitz constant of a neural network, we can bound the degree to which it will be affected by adversarial perturbations. This [paper](https://arxiv.org/abs/1811.05381) does an excellet job of sorting out (pun intended) the details of not only bounding the Lipschitz constant of a neural network, but also hacking the weights to effectively limit the Lipschitz constant of any function it will be able to learn.

## Thresholding Experiments
You'd probably never guess you can cast thresholding as an ML problem. Yup. The problem arises from balancing exploration and exploitation while the model is running in production. Do you set the threshold permanently? If so, there's no way you can monitor false negative rate over time. I ran some experiments in which I used a [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution) to model the probability of true/false positive/negative at any given threshold given the data accumulated to date. As candidate thresholds to chose from, I used existing model scores. I assumed a utility function associated with probability of true/false positive/negatives could be supplied by a business, and attempted to dynamically sample thresholds via [Thompson sampling](https://en.wikipedia.org/wiki/Thompson_sampling). As it turns out, you still end up with biased estimates of probabilities of true/false positives/negatives for each possible threshold if you base these estimates on thresholds greater than 0 (a threshold of 0 includes everything). So my next attempt at solving this problem involved dynamically choosing between setting an optimal threshold, and collecting unbiased data by setting the threshold to 0. Upon choosing the later, you can use that data to obtain a better estimate of the true utility function. I set up an algorithm that attempted to balance the debt incurred from exploring (over exploiting) with the net utility gained over time after exploring and obtaining a better estimate of the utility function. This can actually be done in linear time with respect to the number of model scores obtained so far, but the code to do it was _horrendous_. I actually ended up choosing between exploring and exploiting stochastically to avoid deadlocking. The implementation prescribed choosing exploration according to the probability that the debt incurred from the last round of exploration had been balanced by exploiting the information gained from it.

## Conclusion

This has been a particularly exciting year for me, and I think everything I learned the year before that really paid off in providing a foundation to build on. It seems silly in hindsight, but I was surprised how useful rigorous math, probability theory, and statistics have been. 
