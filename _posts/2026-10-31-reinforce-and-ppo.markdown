---
layout: post
title: "REINFORCE and PPO: When Does One Become the Other?"
---
{% include math.html %}

## Setup

I've been putting off learning [Proximal Policy Optimization (PPO)](https://en.wikipedia.org/wiki/Proximal_policy_optimization), partly because its objective looks substantially more complicated than [REINFORCE](https://en.wikipedia.org/wiki/Policy_gradient_method#REINFORCE). A useful way to make sense of a new algorithm is to ask when it reduces to one we already understand.

For a policy $$\pi_\theta(a|s)$$, the [policy-gradient theorem](https://en.wikipedia.org/wiki/Policy_gradient_method) underlying REINFORCE gives

$$\nabla_\theta J(\theta)=\mathbb E_{\tau\sim\pi_\theta}\left[\sum_t G_t\nabla_\theta\log\pi_\theta(a_t|s_t)\right],$$

where $$G_t$$ is the return from time $$t$$ onward (with appropriate discount conventions). The [score-function (log-derivative) identity](https://en.wikipedia.org/wiki/Score_(statistics)) uses $$\nabla\log\pi=\nabla\pi/\pi$$, which turns a derivative of an expectation into a sampleable expectation. The [law of large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers) justifies approximating this expectation by a sample average; it is not itself an importance-sampling correction.

## Why subtracting a baseline doesn't change the gradient

In REINFORCE we can replace the return $$R(a)$$ with $$R(a)-b(s)$$, where the baseline depends on the state but not the sampled action. Expand the expectation as a sum over actions and cancel the probabilities:

$$
\mathbb E_{a\sim\pi_\theta(\cdot|s)}
[(R(a)-b(s))\nabla_\theta\log\pi_\theta(a|s)]
=
\sum_a(R(a)-b(s))\nabla_\theta\pi_\theta(a|s).
$$

Distribute the terms:

$$
=\sum_a R(a)\nabla_\theta\pi_\theta(a|s)
-b(s)\sum_a\nabla_\theta\pi_\theta(a|s).
$$

**The second term vanishes because the action probabilities always sum to one; therefore their derivatives sum to the derivative of one, which is zero:**

$$
\sum_a\nabla_\theta\pi_\theta(a|s)
=\nabla_\theta\sum_a\pi_\theta(a|s)
=\nabla_\theta 1=0.
$$

Thus the expected gradient is unchanged. Individual sampled gradients *do* change, however, which is where [variance reduction](https://en.wikipedia.org/wiki/Variance_reduction) comes in. The baseline removes a common component of the returns that otherwise contributes noise to finite-sample estimates. A suitable baseline, often $$V^\pi(s)$$, can reduce variance considerably; an arbitrary one is not guaranteed to help. With the value baseline, the expected return minus baseline is the [advantage function](https://en.wikipedia.org/wiki/Proximal_policy_optimization#Policy_gradient_laws:_the_advantage_function), $$A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$$. This statement is about the exact policy-gradient identity; learned or bootstrapped advantages may introduce approximation error.

## PPO's extra ratio

Collect trajectories using $$\pi_{\mathrm{old}}$$, then optimize a new policy against those samples. PPO begins with a surrogate objective using an [importance-sampling](https://en.wikipedia.org/wiki/Importance_sampling) ratio. Unlike the score-function identity above, this ratio reweights actions sampled from the old policy when evaluating the new policy (for the old-policy state distribution):

$$L(\theta)=\mathbb E_{(s,a)\sim\pi_{\mathrm{old}}}\left[r_\theta(s,a)\hat A(s,a)\right],\qquad r_\theta(s,a)=\frac{\pi_\theta(a|s)}{\pi_{\mathrm{old}}(a|s)}.$$

The old policy and estimated advantages are fixed during this optimization. Differentiate:

$$\nabla_\theta L=\mathbb E_{\mathrm{old}}\left[\hat A\,r_\theta\nabla_\theta\log\pi_\theta(a|s)\right].$$

Now set $$\theta=\theta_{\mathrm{old}}$$. The ratio is exactly one, so

$$\left.\nabla_\theta L\right|_{\theta_{\mathrm{old}}}=\mathbb E_{\mathrm{old}}\left[\hat A\nabla_\theta\log\pi_{\mathrm{old}}(a|s)\right].$$

That's the advantage-weighted REINFORCE estimator on the same batch. Because the ratio is one, PPO's clipping is inactive locally too: the initial gradient of the clipped surrogate agrees with this one.

## What changes after the first update?

PPO reuses the sampled trajectories for multiple gradient steps, somewhat like short-lived [experience replay](https://en.wikipedia.org/wiki/Experience_replay). Once the policy moves, $$r_\theta$$ is no longer one. The [clipped surrogate objective](https://en.wikipedia.org/wiki/Proximal_policy_optimization#PPO),

$$L^{\mathrm{CLIP}}=\mathbb E_{\mathrm{old}}\left[\min\left(r_\theta\hat A,\;\operatorname{clip}(r_\theta,1-\epsilon,1+\epsilon)\hat A\right)\right],$$

stops rewarding sufficiently large *favorable* changes in action probability. It does not impose a hard bound on the policy, nor does the per-action ratio exactly correct the changed distribution of whole trajectories.

In that sense PPO reuses experiences for a few approximately [on-policy](https://en.wikipedia.org/wiki/Reinforcement_learning) updates, rather than maintaining a long-lived replay buffer. The core policy gradient hasn't disappeared; PPO wraps it in a surrogate designed to make limited reuse of fresh experience practical.

## References

- [OpenAI Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Schulman et al., Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
