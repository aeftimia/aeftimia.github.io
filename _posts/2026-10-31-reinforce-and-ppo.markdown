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

## Why subtracting a baseline doesn't change the gradient

Suppose we've already derived REINFORCE and want to replace the return with an advantage: return minus some baseline $$b(s)$$. You might reasonably expect this to change the magnitude of the gradient. After all, we're subtracting something from every reward! The trick is to write out the expectation **before** canceling its probability terms.

Fix a state $$s$$ for the moment, and let $$R(a)$$ be the expected return from choosing action $$a$$ there. The expected gradient-ascent update, with learning rate $$\alpha$$, is

$$
\Delta\theta=\alpha\,\mathbb E_{a\sim\pi_\theta(\cdot|s)}
\left[(R(a)-b(s))\nabla_\theta\log\pi_\theta(a|s)\right].
$$

Replace the log derivative with $$\nabla_\theta\pi_\theta(a|s)/\pi_\theta(a|s)$$. Then, rather than canceling anything yet, **expand the expectation into a sum over every possible action**:

$$
\Delta\theta=\alpha\sum_a\pi_\theta(a|s)(R(a)-b(s))
\frac{\nabla_\theta\pi_\theta(a|s)}{\pi_\theta(a|s)}.
$$

Now the probability in the sum cancels the probability in the denominator. But **the sum itself hasn't gone anywhere**:

$$
\Delta\theta=\alpha\sum_a(R(a)-b(s))\nabla_\theta\pi_\theta(a|s).
$$

Distribute and pull out the baseline, which is constant *across actions at this state*:

$$
\Delta\theta=
\alpha\sum_a R(a)\nabla_\theta\pi_\theta(a|s)
-\alpha b(s)\sum_a\nabla_\theta\pi_\theta(a|s).
$$

Here's the important part. The probabilities of **all possible actions sum to one**, for *every* value of $$\theta$$. Therefore the derivative of their sum is the derivative of one:

$$
\sum_a\nabla_\theta\pi_\theta(a|s)
=\nabla_\theta\left(\sum_a\pi_\theta(a|s)\right)
=\nabla_\theta 1=0.
$$

And there it is: the baseline term is $$-\alpha b(s)\times 0$$. It disappears **because we summed the derivatives of every action probability**, not because the baseline was small, or because we adjusted the learning rate. Individual action-probability derivatives need not be zero; their **sum** is zero.

This is an identity for the *expected* gradient, not for the gradient computed from one sampled action. The baseline changes each sampled update and can reduce the variance of those updates without changing their mean. (A badly chosen baseline can increase variance.) Taking $$b(s)=V^\pi(s)$$ gives the usual advantage $$A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$$ in expectation. Approximate learned or bootstrapped advantages bring additional estimation error.

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
