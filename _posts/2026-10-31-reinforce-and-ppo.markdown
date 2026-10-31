---
layout: post
title: "REINFORCE and PPO: When Does One Become the Other?"
---
{% include math.html %}

## Setup

I've been putting off learning [Proximal Policy Optimization (PPO)](https://en.wikipedia.org/wiki/Proximal_policy_optimization). I hadn't had much reason to use it until Unit 2 of [BlueDot Impact's Technical AI Safety course](https://bluedot.org/courses/technical-ai-safety) introduced [reinforcement learning from human feedback (RLHF)](https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback) and [reinforcement learning from AI feedback (RLAIF)](https://en.wikipedia.org/wiki/Reinforcement_learning_from_AI_feedback). PPO's objective looks substantially more complicated than [REINFORCE](https://en.wikipedia.org/wiki/Policy_gradient_method#REINFORCE), so I wanted to work out when one reduces to the other.

For a policy $$\pi_\theta(a|s)$$, the [policy-gradient theorem](https://en.wikipedia.org/wiki/Policy_gradient_method) underlying REINFORCE gives

$$\nabla_\theta J(\theta)=\mathbb E_{\tau\sim\pi_\theta}\left[\sum_t G_t\nabla_\theta\log\pi_\theta(a_t|s_t)\right],$$

where $$G_t$$ is the return from time $$t$$ onward (with appropriate discount conventions). The [score function (log derivative) identity](https://en.wikipedia.org/wiki/Score_(statistics)) uses $$\nabla\log\pi=\nabla\pi/\pi$$, which turns a derivative of an expectation into a sampleable expectation. The [law of large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers) justifies approximating this expectation by a sample average; the sampling distribution remains unchanged.

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

Thus the *expected* gradient is unchanged, but the gradients computed from a finite sample are not. In practice, we generally cannot enumerate every possible action or trajectory, so the baseline contributions that cancel exactly in expectation don't cancel exactly in any given batch. Choosing a suitable baseline [reduces the variance](https://en.wikipedia.org/wiki/Variance_reduction) of the gradient estimator from finite samples, giving us a more reliable estimate from the same data. Discovering rare actions with high rewards remains a separate exploration problem.

A common choice is the [value function](https://en.wikipedia.org/wiki/State-value_function) $V^\pi(s)$, giving the [advantage function](https://en.wikipedia.org/wiki/Advantage_function) $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ in expectation. The baseline need not be exactly the expected return to preserve the identity, and a poorly chosen one can increase variance. Learned or bootstrapped advantage estimates introduce their own approximation issues.

## Reusing expensive trajectories

A more reliable estimate still requires collecting experience. Sometimes that is cheap; sometimes each trajectory means interacting with an environment, running a simulator, or generating and scoring a long [language model](https://en.wikipedia.org/wiki/Large_language_model) completion. A baseline reduces the noise in each gradient estimate. PPO makes several optimization steps using each recent batch. [Exploration](https://en.wikipedia.org/wiki/Exploration-exploitation_dilemma) presents another challenge: finding useful actions in the first place.

## PPO's extra ratio

Collect trajectories using $$\pi_{\mathrm{old}}$$, then optimize a new policy against those samples. PPO begins with a surrogate objective using an [importance sampling](https://en.wikipedia.org/wiki/Importance_sampling) ratio. This ratio reweights actions sampled from the old policy while retaining its state distribution:

$$L(\theta)=\mathbb E_{(s,a)\sim\pi_{\mathrm{old}}}\left[r_\theta(s,a)\hat A(s,a)\right],\qquad r_\theta(s,a)=\frac{\pi_\theta(a|s)}{\pi_{\mathrm{old}}(a|s)}.$$

The old policy and estimated advantages are fixed during this optimization. Differentiate:

$$\nabla_\theta L=\mathbb E_{\mathrm{old}}\left[\hat A\,r_\theta\nabla_\theta\log\pi_\theta(a|s)\right].$$

Now set $$\theta=\theta_{\mathrm{old}}$$. The ratio is exactly one, so

$$\left.\nabla_\theta L\right|_{\theta_{\mathrm{old}}}=\mathbb E_{\mathrm{old}}\left[\hat A\nabla_\theta\log\pi_{\mathrm{old}}(a|s)\right].$$

Because the ratio is one, PPO's clipping is inactive locally. **At the old policy, the clipped PPO gradient equals the REINFORCE gradient weighted by advantage on the same batch.**

## What changes after the first update?

PPO reuses the sampled trajectories for multiple gradient steps, somewhat like short term [experience replay](https://en.wikipedia.org/wiki/Experience_replay). Once the policy moves, $$r_\theta$$ is no longer one. The [clipped surrogate objective](https://en.wikipedia.org/wiki/Proximal_policy_optimization#PPO),

$$L^{\mathrm{CLIP}}=\mathbb E_{\mathrm{old}}\left[\min\left(r_\theta\hat A,\;\operatorname{clip}(r_\theta,1-\epsilon,1+\epsilon)\hat A\right)\right],$$

stops rewarding sufficiently large *favorable* changes in action probability. Clipping discourages large updates without imposing a hard bound on the policy. The action probability ratio also leaves shifts in the state distribution uncorrected.

In that sense PPO reuses experiences for a few approximately [on-policy](https://en.wikipedia.org/wiki/Reinforcement_learning) updates, and then collects fresh samples instead of maintaining an extensive replay buffer. The clipped surrogate allows several updates on fresh experience before collecting another batch.

## References

- [OpenAI Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Schulman et al., Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
