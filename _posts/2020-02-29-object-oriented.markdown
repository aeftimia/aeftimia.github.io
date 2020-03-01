---
layout: post
title:  "Object Oriented Programming"
---
{% include math.html %}

I've been a diehard functional programmer for the better part of 10 years. Before I took up Python in 2011, I had been programming in Mathematica for about 4 years, and it left a strong impact on how I think about code. I generally write small scripts and, on rare occasion, modest (~800 lines) frameworks that process data and/or do something mathy. I take pride in minimizing redundancy of my code without seriously impacting readability. I have spent the last three months working on an R&D project at work focused on monitoring machine learning models in production, and sending alerts when things seem off. Among the artifacts I'll be providing the company from my work is a fully documented Python library for calculating statistical [divergences](https://en.wikipedia.org/wiki/Divergence_(statistics). While this project has probably been the highlight of my last 5 working years, I'll save a rant about it for another post.

So where was I? Oh right! Die hard funcional programmer! Right, so as a functional programmer, I naturally structured my divergence library around a handful functions that compose in different ways. The functions I defined are fairly general, and necessarily have a bunch of optional arguments to override defaults. For example, some train neural networks, some control the data sampling process during trainng and validation, some control neural network hyperparameters, etc. A large number of them are fed into one big function. We'll call this `F`.

```
def F(..., a=..., b=..., c=...):
    return G(..., a) * H(..., b) * K(..., c))
```

This obviously isn't the code itself, but it's a good illustration of the situation. I want the user to be _able_ to change the default arguments of `G`, `H`, and `K`. To do this, I have to keep adding them as arguments to `F` with some default parameters. This process works alright, but it can be tedeous to keep refactoring `F` every time `G` or `H` change. Basically, you have to change everything twice. And yes, I really do want `G` and `H` separate because it's useful to be able to replace them with different functions to create new versions of `F`. Let's just assume I've thought out the breakdown well enough that this is probably the optimal structure for minimal redundancy.

A smart friend suggested having `F` just take a `**kwargs` that includes everything that could ever be passed to `G`, `H`, and `K`, and then have those respective functions ignore irrelavent `kwargs`. This is slick and actually fits my situation because the few arguments shared between `G`, `H`, and `K` really _should_ be the same! Aside from potential scaling issues (do I really want everyone working on this project to have to coordinate naming conventions based on how functions are composed?), this trick only works if `G`, `H`, and `K` all take `**kwargs` in their function head so they can ignore unneeded `**kwargs`.

There actually _is_ an efficient solution to this problem. 

```
class F:
    G_defaults = {}
    H_defaults = {}
    K_defaults = {}
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for name, new_kwargs in kwargs.items():
            if hasattr(self, name):
                self.__getattr__(name).update(new_kwargs)

    def __call__(self, *args, **kwargs):
        if kwargs:
            return self.__class__(**self.kwargs).__init__(**kwargs)(*args)
        return G(..., **self.G_defaults) * H(..., **self.H_defaults) * K(..., **self.K_defaults))
```

Now I can initialize `F` with separate `**kwargs` directed at each function it will call *and* efficiently encode defaults! Note this isn't _quite_ inheritence because inheritence would _override_ one or more of `G_defaults`, `H_defaults`, and `K_defaults`. Inheritance wouldn't be bad, but we can do one better with a nicely crafted `__init__` function. Now I can specify _exactly_ what's different in the default arguments and never have to worry about naming conflicts.
