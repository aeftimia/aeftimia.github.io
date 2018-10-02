---
layout: post
title:  "Submodular Optimization and Machine Learning"
---
{% include math.html %}

## Introduction

I found a great post on [submodular optimization](https://jeremykun.com/2014/07/07/when-greedy-algorithms-are-good-enough-submodularity-and-the-1-1e-approximation/) curtosy of Jeremy Kun. The larger rule for submoular optimization that given a function $$f$$ defined on subsets of a given set that is both monotone, $$f\left(A\cup B\right)\ge f\left(A\right)$$, and submodular $$f\left(A\cup B\cup\left\{y\right\}\right) - f\left(A\cup B\right) \le f\left(A\cup \left\{y\right\}\right) - f\left(A\right)$$, greedy maximization of $$f$$ given a cardinality constraint (the argument must have at most $$k$$ elements) will achieve at least $$1 - 1 / e$$ the global maximum value of $$f$$ over any set that satisfies the cardinality constraint. If this isn't 100% obvious to you, go read Jeremy Kun's blog. I'll wait!

OK, it's obvious now, right? Great! Because when you apply this to entropy as a function of features in a given feature set, the rule has an important application to information theory and machine learning that I thought was worth working through.

## Notation

It was surprisingly tricky to develop notation that is both consistent and leads to equations that fit on one line. I ended up using a few shorthands that may or may not be subject to misinterpretation, so I'll outline everything here. We'll say $$A$$ and $$B$$ are sets of features. For instance, if we were defining features on cars, we might have "color", "frame size", and "manufacturer".  I use upper case letters for sets of features and lowercase letters for potential values for those features. For instance, if $$A$$ was the set of features "color" and "manufacturer", $$a\in A$$ might be (green, Honda), and belong to someone with really bad tastes.

The astute reader is probably whining that I'm implicitly using upper case letters for both sets of features and Cartesian products of the elements of those features. Since pretty much every calculus texts butchers notation a lot worse than this, I'm going to assume we're all adults here (or at least have the maturity expected of one) and can move on with our lives once we both understand what the other is trying to say.

Now that you're content with that, I'm going to butcher some more notation.

Shorthand for joint distributions over several subsets of features:

$$p\left(a, b, \dots\right) = p\left(a_1,\dots,a_\left\|A\right\|,b_1,\dots,b_\left\|B\right\|,\dots\right)$$

Shorthand for entropy over several subsets of features:

$$ H(A\cup B) = -\sum_{a\in A, b\in B} p\left(a, b\right)\log\left(p\left(a, b\right)\right) $$

I'm going with set builder notation over outer products here to help map common notions of probability distributions and entropy onto a notation that describes functions of sets. I originally tried this with sigma algebras, but the proofs went way off the page. I'm unfortunately substituting a visual eyesore for a mathematical one because if the visual eyesore prevents otherwise mathematically literate readers from getting through the damn post, then adhering to the traditional use of mathematical operators popular enough to make it into the MathJax renderer is doing more harm than good.

## Monotonicity

$$f$$ is considered _monotone_ if

$$f(B) \le f(A)\forall B\subseteq A$$

First, let's show that entropy is monotone.

$$H\left(A \cup \left\{Y\right\}\right) = -\sum_{a\in A, y\in Y} p\left(a, y\right)\log\left(p\left(a, y\right)\right)$$
$$= -\sum_{a\in A, y\in Y} p\left(a, y\right)\log\left(p\left(a, y\right)\right)$$

Note that 

$$p\left(a\right) = \sum_{y\in Y}p\left(a, y\right) \ge p\left(a, y\right)\forall y\in Y$$

Therefore,

$$H\left(A \cup \left\{Y\right\}\right) = -\sum_{a\in A, y\in Y} p\left(a, y\right)\log\left(p\left(a, y\right)\right) \le -\sum_{a\in A, y\in Y} p\left(a\right)\log\left(p\left(a\right)\right) = H\left(A \cup \left\{Y\right\}\right) $$

Q.E.D.

## Submodularity

Next, we'll show $$H$$ is submodular. For this proof, we'll use the following variation on the definition of submodularity:

$$f\left(A\cup B \cup \left\{Y\right\} \right) - f\left(A\cup B \right) \le f\left(A\cup \left\{Y\right\} \right) - f\left(A \right) $$

Applying the left hand side to the entropy function, we find

$$H\left(A\cup B \cup \left\{Y\right\} \right) - H\left(A\cup B \right) = -\sum_{a\in A, b\in B, y\in Y}p\left(a,b,y\right)\log\left(p\left(a,b,y\right)\right) + \sum_{a\in A, b\in B} p\left(a, b\right)\log\left(p\left(a, b\right)\right) $$

If we do the same for the right hand side,

$$ -\sum_{a\in A,y\in Y}p\left(a,y\right)\log\left(p\left(a,y\right)\right) + \sum_{a\in A} p\left(a\right)\log\left(p\left(a\right)\right) $$

Now we note that $$\sum_{b\in B} p\left(a,b,y\right) = p\left(a,y\right)$$ and $$\sum_{b\in B} p\left(a,b\right) = p\left(a\right)$$ and make respective substitutions on the right hand side _outside_ the logarithms.

$$ -\sum_{a\in A,b\in B,y\in Y}p\left(a,b,y\right)\log\left(\frac{p\left(a,y\right)}{p\left(a\right)}\right)$$

Let's try proof by contradiction so we can compare the new left and right hand sides and try to simplify.

$$ -\sum_{a\in A,b\in B,y\in Y}p\left(a,b,y\right)\log\left(\frac{p\left(a,b,y\right)}{p\left(a,b\right)}\right) > -\sum_{a\in A,b\in B,y\in Y}p\left(a,b,y\right)\log\left(\frac{p\left(a,y\right)}{p\left(a\right)}\right)$$

Lifting everything through the logarithms,

$$ \frac{p\left(a,b,y\right)}{p\left(a,b\right)} > \frac{p\left(a,y\right)}{p\left(a\right)}$$

Now let's multiply out the denominators.

$$ p\left(a,b,y\right)p\left(a\right) > p\left(a,y\right)p\left(a,b\right)$$

Now let's sum over $$b\in B$$.

$$ \sum_{b\in B} p\left(a,b,y\right)p\left(a\right) = p\left(a,y\right)p\left(a\right) > \sum_{b\in B} p\left(a,y\right)p\left(a,b\right) = p\left(a,y\right)p\left(a\right)$$

Hence a contradiction, proving $$H$$ submodular.

**Q.E.D.**

## Implications
We've now shown entropy is monotone and submodular. The most straightforward result is that greedily building decision trees to maximize entropy gained from subsets of $$k$$ features will converge to a decent approximation (namely at least $$1 - 1 / e \approx 63\%$$) of the highest entropy $$k$$ element subset of features. In other words, you'll always get at least $$63\%$$ of the entropy you'd achieve from the best decision tree given a maximum tree depth. In practice, this limit can be made even higher if you keep running the algorithm after you already hit your prescribed tree depth, and at each step swap out your worst feature for the next best one that isn't in your tree yet.

There are, however, broader implications. Decision trees are a very simple example of building predictive models from subsets of the data's feature space. However, the general idea of preprocessing data by pruning covariant or otherwise unnecessary features will generally improve any classification algorithm. The overarching rule here is that quality feature selection can be done greedily.
