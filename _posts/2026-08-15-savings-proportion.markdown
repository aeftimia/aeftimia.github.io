---
layout: post
title:  "Retirement"
---
{% include math.html %}

I tend to only write about things that change the way I look at things. Maybe it's just getting older, but that hasn't happened in a while. I've taken up interests outside math and programming, but I like to apply my professional experience to my day to day interests. In particular, my wife and I have talked a lot about personal finance and the possibility of early retirement.

I went into data science to apply mathematics to useful problems. It's something I truly enjoy, and I never really cared that much about retiring (though I certainly contributed to my 401k so I'd have the option). Then, a few years back, I read [Die with Zero](https://diewithzerobook.com/). It was really interesting. Some of his argument for retiring when you can is simply that even people who love what they do would probably love doing something else more. His stronger argument for why you should at least try to die with nothing left is more or less that you can't take it with you. He applies that reasoning to some dancer friends who truly would prefer dancing to pretty much anything else. He also talked about how money is worth less as you get older. Not because of inflation, but because if you only have a few years left, an extra million in the bank just doesn't help much. My wife has done some ruthless audits of our spending. While we're not cheap, we're at least conscientiously lean now. She realized early retirement was on the table if we played our cards right.

Since then, we've both thought of friends who could probably do the same. Some make far less, but just spend way less too. It got me thinking about whether there was a concise argument for saving some proportion of your income for early retirement. The clearest one seems to look like this: suppose your index/mutual/fairly safe investment of choice yields 10% per year. I know there are going to be about 647 people who balk and say the stock market doesn't do that well. To which I'd first say, yes it almost always does, and in fact does [better in recent years](https://www.fool.com/investing/how-to-invest/index-funds/average-return/). But even *that's beside the point*! You could just call your expected ROI $$r$$ and continue without nitpicking how irrationally pessimistic you are about long and short term historical averages predicting future returns. Anyway, to make the numbers easy, if we say 10% a year ROI, then in 7 years of investing 50% of your take home, you actually end up with 5x that same take home invested. At this point, your _expected_ yield in interest equals the other 50% you've been spending to get there. This, by the way, comes out to be about 9 years if you assume 2% inflation. I'm not saying you'd necessarily want to pull the trigger at this point since we all know that 10% isn't guaranteed, but I definitely think it gets you a ballpark for how long it would take before you could retire with a similar quality of life. Especially if you're not afraid to go live in a poor country for a bad year or go back to work if there's a recession. The neat thing about this argument is I never assumed anything about your income or lifestyle. I think this applies to a lot of dual income couples.

The more general rule looks like this. Say you take home income $$I$$ per year, and you save a fraction $$s$$ of it, investing the rest at annual return $$r$$. Each year you contribute $$sI$$ to the pile, and it compounds. After $$t$$ years, the future value of those contributions (an ordinary annuity) is

$$
P(t) = sI \cdot \frac{(1+r)^t - 1}{r}
$$

I'll call myself "retired" the moment the portfolio's *investment income alone* covers my current spending, $$(1-s)I$$, without touching principal. That is when

$$
r \cdot P(t) = (1-s)I.
$$

Substituting and canceling $I$:

$$
s\left[(1+r)^t - 1\right] = 1 - s \quad\Longrightarrow\quad (1+r)^t = \frac{1}{s}
$$

$$
\boxed{t(s) = \frac{\ln(1/s)}{\ln(1+r)} = \frac{-\ln s}{\ln(1+r)}}
$$

That's a strikingly clean result: time to retirement depends only on your savings rate $$s$$ and your return $$r$$. It has nothing (directly) to do with your income or your lifestyle. Plug in $$s = 0.5$$ and $$r = 10\%$$:

$$
t = \frac{\ln 2}{\ln 1.1} \approx \frac{0.693}{0.0953} \approx 7.3 \text{ years}
$$

Hence the "save half, retire in ~7 years" rule of thumb. If you knock $$r$$ down to inflation-adjusted return, using $$2\%$$ inflation gives $$r_{\text{real}} = \frac{1.10}{1.02} - 1 \approx 7.84\%$$, and

$$
t = \frac{\ln 2}{\ln 1.0784} \approx 9.2 \text{ years,}
$$

matching the "closer to 9 years once you account for inflation" intuition. I'm not saying you'd necessarily want to pull the trigger at exactly $$t(s)$$, since $$r$$ isn't guaranteed — but it gives you a solid ballpark, especially if you're not afraid to ride out a bad year abroad or go back to work after a recession. The neat thing about this argument is that it never assumed anything about your income or lifestyle — just $$s$$ and $$r$$. I think that generalizes well to a lot of dual income couples with pretty different paychecks.

Just an aside here for anyone that gets all squirmy thinking of retiring the minute their expected interest equals their expected expenses. We can introduce a margin $$m \geq 1$$ for how many times over you want your interest to cover your spending before saying you could retire. The condition becomes $$rP(t) = m\cdot(1-s)I$$ instead of exact equality (so if $$m=1.5$$, the "5x take home" target from earlier just becomes "7.5x take home"), which reshapes the result above to

$$
t(s) = \frac{\ln\!\left(\dfrac{m}{s} - (m-1)\right)}{\ln(1+r)}
$$

This is the same $$1/s$$ from the plain rule, just scaled by $$m$$ and shifted down by a constant $$(m-1)$$. The point is $$t(s)$$ has the same shape and remains invariant with your income $$I$$ even accounting for some safety margin.

Today I noticed something else about this, and maybe it's almost as useful. As my wife and I considered a move to a higher cost-of-living area in exchange for more money, I asked whether our new expenses would hurt our early-retirement timeline. I went back to $$t(s)$$. If a move raises both your income and your spending but leaves $$s$$ unchanged, your timeline to retirement doesn't move at all — and the lifestyle you retire *into* scales up right along with the lifestyle you had while saving, because your target portfolio $$P^* = m \cdot (1-s)I/r$$ scales linearly with $$I$$. As long as you don't let $$s$$ slip, you get to carry the upgrade with you: retire on the same schedule, spend more getting there, and spend more once you're there.
