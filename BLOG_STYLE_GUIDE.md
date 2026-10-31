# Blogging style and editorial guide

Notes from the October 31, 2026 REINFORCE-to-PPO drafting conversation. Use when helping author or edit future posts on aeftimia.github.io.

## Voice and audience
- Model tone on the older posts, particularly *Submodular Optimization and Machine Learning* (2018), *Stop Imputing Nulls?* (2020), and *Object Oriented Programming* (2020).
- Write as a mathematically literate practitioner working something out, not as a textbook instructor or marketing writer. Conversational, occasionally playful, direct, compact. Respect readers' mathematical competence.
- Start with an interesting concrete question, identity, or reduction, and end with what it implies.
- Favor insights that show **when one technique reduces to another**, identifying exact conditions and caveats. This is an important way the author consolidates understanding.
- **Connect concepts into a coherent mental model.** Show how ideas from different papers, algorithms, and eras relate, where they share mathematical structure, and where they diverge. A strong post helps readers *derive and relate* ideas instead of memorizing disconnected definitions. Prefer illuminating conceptual links over a catalog of facts, while retaining precision about the limits of each analogy.
- Keep the article reasonably short. Avoid over-explaining routine steps or repeating a point once the key mathematical identity is established.

## Mathematics and conceptual precision
- Derive equations when their algebra is the central insight; don't derive standard machinery just because it is available.
- Be explicit about **where a cancellation occurs** (e.g., expand an expectation into a sum before canceling probabilities; the sum of action-probability derivatives is zero because probabilities sum to one for every parameter value).
- Maintain clear distinctions: population expectation vs finite-sample estimate; bias vs variance; variance reduction vs sample reuse vs exploration; policy action importance ratios vs score-function log derivatives.
- Use standard terminology such as **variance** naturally rather than euphemisms like “wobble.” Explain what variance reduction buys the practitioner, without a detour into its historical definition or a derivation of variance unless specifically needed.
- Ground motivation in real sample limitations, including high-dimensional or large action/trajectory spaces and expensive rollouts. Don't imply action-space cardinality *by itself* determines Monte Carlo error, or that a baseline helps discover rare high-reward actions.
- Use examples only if they illuminate an issue. Tiny two-action examples can obscure why sampling is costly when all actions are trivially enumerable. A short general derivation may be better.
- Don't equate similar but distinct algorithms or overclaim exactness: PPO's first gradient at old=new matches advantage-weighted REINFORCE (holding advantages fixed and under the same data/conventions), not necessarily the entire algorithm. PPO is classified as on-policy despite short-term reuse, and per-action ratios aren't full trajectory-distribution corrections.
- Caveat estimates/assumptions when material; avoid piling on disclaimers.

## Copy editing and sentence construction
- Avoid inflated or formulaic AI phrasing such as “a striking result,” “the interesting thing is,” “that kind of association,” and “risks we already have some intuition for” when a direct concrete statement will do. State the actual comparison, number, or implication without announcing its significance. Favor short, natural sentences over rhetorical setup.
- Prefer active voice. Identify the actor and use a direct verb whenever possible; keep passive voice when the agent is irrelevant or it improves precision.
- Avoid reflexive “not X, but Y,” “for X rather than Y,” and “this was done for X, not Y” constructions. They often add length and create an artificial contrast. State the positive claim directly. Use contrast only when correcting a meaningful misconception.
- Remove unnecessary hyphens from conversational expressions; keep standard technical compounds when they prevent ambiguity (e.g., “on-policy,” “state-dependent,” “finite-sample” as modifiers).
- Make a dedicated last pass for active voice, needless contrast, unnecessary hyphenation, and wordiness. Read it aloud.

## Hyperlinks and citations
- Wikipedia-inspired, concept-dense linking is a strong stylistic preference, reminiscent of Gwern: link important concepts on first occurrence inline, especially established technical terms (law of large numbers, importance sampling, REINFORCE, policy gradients, variance reduction, advantages, experience replay, etc.).
- Prefer relevant Wikipedia pages for general concepts; original papers and authoritative technical sources for algorithms and nontrivial claims.
- Check that a link actually resolves to the intended concept; avoid guessed Wikipedia slugs or misleading anchors.
- Avoid repeated links and link clutter; keep a short references section for original papers/major sources.
- Before finishing, do a dedicated hyperlink pass.

## Efficient drafting workflow
1. Read a couple of older source posts to recover voice, then draft; don't repeatedly ask about stylistic preferences already stated here.
2. Establish the main *reduction/identity* and conditions early. Structure around the author's question, not a broad survey.
3. Audit algebra and the exact scope of identities; distinguish what holds in expectation versus per sample.
4. Cut wordiness: replace lengthy narration with a tight derivation and one sentence on why it matters.
5. Check practical motivation and keep distinct problems separate (variance reduction, replay/reuse, exploration).
6. Make hyperlink and copy editing passes; check active voice, unnecessary contrast, hyphenation, references/targets, and rendered math/Markdown formatting.
7. Read the draft aloud if requested and iterate on specific passages.
8. Preserve unpublished work on a separate GitHub branch; never publish or merge without explicit request.

## Current working example
- Branch: `draft/reinforce-ppo`
- Draft: `_posts/2026-10-31-reinforce-and-ppo.markdown`
- Favorite teaching move discovered in the discussion: expand the policy expectation into a sum, distribute the baseline, observe `sum_a ∇θ πθ(a|s) = ∇θ 1 = 0`. The key point needs just a sentence once the algebra is written.
