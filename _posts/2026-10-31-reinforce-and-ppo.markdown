---
layout: post
title: "REINFORCE and PPO: When Does One Become the Other?"
---
{% include math.html %}

## Setup

I've been putting off learning Proximal Policy Optimization (PPO), partly because its objective looks substantially more complicated than REINFORCE. A useful way to make sense of a new algorithm is to ask when it reduces to one we already understand.

For a policy $$\pi_\theta(a|s)$$, the policy-gradient identity behind REINFORCE is

$$\nabla_\theta J(\theta)=\mathbb E_{\tau\sim\pi_\theta}\left[\sum_t G_t\nabla_\theta\log\pi_\theta(a_t|s_t)\right],$$

where $$G_t$$ is the return from time $$t$$ onward (with appropriate discount conventions). The log derivative is just the identity $$\nabla\log\pi=\nabla\pi/\pi$$, which lets us estimate a derivative of an expectation by sampling from the policy.

## A baseline doesn't change the expected gradient

We can replace $$G_t$$ by $$G_t-b(s_t)$$, so long as the baseline doesn't depend on the sampled action. To see why, condition on a state:

$$\mathbb E_{a\sim\pi_\theta(\cdot|s)}[b(s)\nabla_\theta\log\pi_\theta(a|s)]$$

$$=b(s)\sum_a\nabla_\theta\pi_\theta(a|s)=b(s)\nabla_\theta 1=0.$$

The cancellation deserves a closer look because it's easy to miss a sum when moving between expectations and log derivatives. Start with the baseline-adjusted *expected* update (learning rate $\alpha$):

$\Delta\theta=\alpha\,\mathbb E_{a\sim\pi_\theta(\cdot|s)}\left[(R(a)-b(s))\frac{\nabla_\theta\pi_\theta(a|s)}{\pi_\theta(a|s)}\right].$

**Expand the expectation as a sum over every possible action before canceling anything:**

$\Delta\theta=\alpha\sum_a\pi_\theta(a|s)(R(a)-b(s))\frac{\nabla_\theta\pi_\theta(a|s)}{\pi_\theta(a|s)}.$

Cancel the probabilities. **The sum over actions remains:**

$\Delta\theta=\alpha\sum_a(R(a)-b(s))\nabla_\theta\pi_\theta(a|s).$

Now split the sum and pull the action-independent baseline out:

$\Delta\theta=\alpha\sum_a R(a)\nabla_\theta\pi_\theta(a|s)-\alpha b(s)\sum_a\nabla_\theta\pi_\theta(a|s).$

Because action probabilities sum to one for *every* value of $\theta$,

$\sum_a\nabla_\theta\pi_\theta(a|s)=\nabla_\theta\left(\sum_a\pi_\theta(a|s)\right)=\nabla_\theta 1=0.$

So the baseline contribution is exactly zero **after summing over actions**, even though it is generally nonzero for any *one sampled action*. This is the part that's easy to lose by mentally canceling the sampling probability and forgetting that the expectation was a sum.

The baseline changes each sampled gradient but leaves its expectation unchanged. It often reduces variance, although a poorly chosen baseline can increase it. Taking $$b(s)=V^\pi(s)$$ gives an advantage in expectation, $$A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$$. A learned or bootstrapped advantage estimate introduces its own approximation issues.

## PPO's extra ratio

Collect trajectories using $$\pi_{\mathrm{old}}$$, then optimize a new policy against those samples. PPO begins with an importance-weighted surrogate:

$$L(\theta)=\mathbb E_{(s,a)\sim\pi_{\mathrm{old}}}\left[r_\theta(s,a)\hat A(s,a)\right],\qquad r_\theta(s,a)=\frac{\pi_\theta(a|s)}{\pi_{\mathrm{old}}(a|s)}.$$

The old policy and estimated advantages are fixed during this optimization. Differentiate:

$$\nabla_\theta L=\mathbb E_{\mathrm{old}}\left[\hat A\,r_\theta\nabla_\theta\log\pi_\theta(a|s)\right].$$

Now set $$\theta=\theta_{\mathrm{old}}$$. The ratio is exactly one, so

$$\left.\nabla_\theta L\right|_{\theta_{\mathrm{old}}}=\mathbb E_{\mathrm{old}}\left[\hat A\nabla_\theta\log\pi_{\mathrm{old}}(a|s)\right].$$

That's the advantage-weighted REINFORCE estimator on the same batch. Because the ratio is one, PPO's clipping is inactive locally too: the initial gradient of the clipped surrogate agrees with this one.

## What changes after the first update?

PPO reuses the sampled trajectories for multiple gradient steps. Once the policy moves, $$r_\theta$$ is no longer one. The clipped objective,

$$L^{\mathrm{CLIP}}=\mathbb E_{\mathrm{old}}\left[\min\left(r_\theta\hat A,\;\operatorname{clip}(r_\theta,1-\epsilon,1+\epsilon)\hat A\right)\right],$$

stops rewarding sufficiently large *favorable* changes in action probability. It does not impose a hard bound on the policy, nor does the per-action ratio exactly correct the changed distribution of whole trajectories.

In that sense PPO reuses experiences for a few approximately on-policy updates, rather than maintaining a long-lived replay buffer. The core policy gradient hasn't disappeared; PPO wraps it in a surrogate designed to make limited reuse of fresh experience practical.

## References

- [OpenAI Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Schulman et al., Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
