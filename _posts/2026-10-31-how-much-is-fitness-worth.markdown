---
layout: post
title: "How Much Is Fitness Worth?"
---

Reading Peter Attia's [*Outlive*](https://www.penguinrandomhouse.com/books/705161/outlive-by-peter-attia-md-with-bill-gifford/), I was surprised by how strongly physical fitness predicts mortality.

In a [study of 122,007 adults](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2707428), participants in the 25th–49th percentiles of cardiorespiratory fitness had 1.41 times the adjusted all-cause mortality hazard of those in the 50th–74th percentiles. The study reported exactly the same hazard ratio, 1.41, for smoking.

Even at the highest fitness levels, the association persisted. Participants in the top 2.3% had **23% lower mortality hazard** than those in the 75th–97.6th percentiles. That's roughly **80% of the proportional mortality hazard reduction associated with being a nonsmoker rather than a smoker** in the same study. The researchers found no apparent upper limit to the association between fitness and longevity.

These findings have been [replicated across many studies](https://pmc.ncbi.nlm.nih.gov/articles/PMC11103301/), often adjusting for age, sex, smoking, comorbidities, and other confounders. A 2024 review summarized 199 unique cohort studies representing more than 20 million observations. The association also appears with [grip strength](https://www.bmj.com/content/361/bmj.k1651), another useful measure of general fitness.

One reason this caught my attention was that I'd been trying to encourage a close friend to take his health more seriously. He understood perfectly well why smoking was dangerous, but was skeptical that fitness mattered nearly as much. Encouraging someone to quit smoking is generally considered sensible, yet encouraging friends to become more physically fit seems considerably less common.

I wanted to understand what the research meant in absolute terms.

## From Fitness Percentiles to Years of Life

Most studies report [hazard ratios](https://en.wikipedia.org/wiki/Hazard_ratio), sometimes comparing fitness percentiles within age and sex groups. These establish relative differences in mortality, but they don't immediately tell us how much longer someone might expect to live or how much their VO₂ max would have to improve.

So I built a [fitness and mortality calculator](https://aeftimia.github.io/fitness-mortality/).

The model combines [FRIEND 2022](https://doi.org/10.1016/j.mayocp.2021.08.020) reference distributions of VO₂ max, [Kokkinos et al.](https://doi.org/10.1016/j.jacc.2022.05.031) estimates of mortality hazard per unit of fitness, and [US life tables](https://www.ssa.gov/oact/STATS/table4c6.html), which give baseline mortality rates by age and sex. It applies a similar approach to grip strength, using [international grip strength norms](https://pmc.ncbi.nlm.nih.gov/articles/PMC11863340/) and [published mortality associations](https://www.bmj.com/content/361/bmj.k1651).

The important step is normalizing the fitness hazard model against the population fitness distribution at each age and sex. This preserves the population mortality rates in the life tables while estimating how mortality varies across fitness levels.

That makes it possible to express the results in absolute probabilities, years of life expectancy, and equivalent risks from other activities.

VO₂ max and grip strength are useful proxies for general physical fitness. As Peter Attia describes them, they're integrators of the work required to achieve a given level of fitness, often accumulated over years. Squeezing a grip strengthener presumably won't reproduce the health benefits associated with being physically strong, though rock climbers might be an interesting exception.

## Risk Equivalence

My friend and I wanted a more intuitive way to compare mortality risks. He's someone I can be brutally honest with, so we could discuss the risks of poor fitness without worrying about offending each other.

He suggested [BASE jumping](https://en.wikipedia.org/wiki/BASE_jumping) as a reference point.

A [study of 20,850 BASE jumps](https://pubmed.ncbi.nlm.nih.gov/17495709/) recorded nine fatalities, or approximately one death per 2,317 jumps. That's about 430 [micromorts](https://en.wikipedia.org/wiki/Micromort), where one micromort represents a one in a million probability of death.

I liked the comparison because it gave us a concrete unit for discussing otherwise abstract differences in annual mortality risk. We could express the additional risk associated with lower fitness as a number of BASE jumps per year.

The comparison also turned out to be useful in an entirely different context.

Mountain biking is one of my hobbies, and people frequently tell me how dangerous they think it is. I've noticed that the people most concerned often aren't particularly athletic themselves. It's tempting to point out that sitting on your ass all day may carry a greater mortality risk than mountain biking, although that doesn't make for particularly polite dinner conversation.

A more diplomatic response is to ask how many miles are on someone's odometer.

Using recent [US traffic fatality statistics](https://www.nhtsa.gov/press-releases/nhtsa-estimates-39345-traffic-fatalities-2024), driving roughly 40,000 miles carries a fatality risk comparable to a BASE jump. Most people accept that risk routinely.

The [FDA](https://www.fda.gov/radiation-emitting-products/medical-x-ray-imaging/what-are-radiation-risks-ct) also estimates that a **chest CT scan** delivering approximately 10 millisieverts may add a one in 2,000 lifetime probability of fatal cancer.

These risks are all in the neighborhood of **half a millimort**, or 500 micromorts.

They give us a common scale for comparing activities that people perceive very differently.

## A Concrete Example

Consider a 34-year-old man moving from median VO₂ max to the 75th percentile for his age.

Using the calculator's age and sex adjusted fitness distribution:

| | 50th percentile | 75th percentile |
|---|---:|---:|
| VO₂ max | 42.5 | 49.0 |
| Annual mortality probability | 0.258% | 0.195% |
| Estimated remaining life expectancy | 43.1 years | 46.3 years |

That's an increase of about **6.5 mL/kg/min in VO₂ max**, corresponding to a 24% lower mortality hazard.

The model associates that difference with approximately **3.3 additional years of life expectancy**.

At age 34, the corresponding annual mortality difference is about 0.063 percentage points, or **630 micromorts per year**.

Expressed using our earlier reference points, that's equivalent to approximately **1.5 BASE jumps or 58,000 miles of driving annually**.

## What If You Hate Exercising?

The remaining question is how much exercise it actually takes to improve fitness by that amount.

The [HERITAGE Family Study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9012529/) provides an unusually useful example. Previously inactive adults completed 20 weeks of supervised cycling, three sessions per week. Sessions progressed from 30 minutes to 50 minutes, with increasing intensity.

Among men aged 17–35, the average improvement in VO₂ max was **15.7%**.

Our hypothetical improvement from the 50th to the 75th percentile requires about **15.3%**.

That suggests that a fairly ordinary exercise commitment, eventually reaching 2.5 hours of cycling per week, can produce an improvement of roughly the magnitude we're considering. Individual responses varied considerably, and the study measured improvements over 20 weeks rather than the exercise required to maintain them indefinitely.

At 2.5 hours per week, someone would spend approximately 130 hours exercising each year.

Over 40 years, that's 5,200 hours, or about 217 days of exercise.

Compare that with the **3.3 additional years**, or roughly 1,200 days, associated with the higher fitness level in the mortality model.

The implied return is about **five and a half days of additional expected life for each day spent exercising**.

That calculation assumes the fitness difference persists and that improving fitness produces the mortality benefit associated with it. The precise causal effect remains uncertain. Exercise trials establish that fitness is modifiable, while large observational studies establish the relationship between fitness and mortality.

I find the comparison useful because even a substantially smaller causal benefit could justify a few hours of exercise each week.

And that calculation assigns no value to the exercise itself, the improved physical capabilities, or the years spent in better health.
