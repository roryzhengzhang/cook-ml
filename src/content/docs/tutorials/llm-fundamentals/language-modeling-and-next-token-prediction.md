---
title: "Language Modeling"
description: "How a model predicts the next token, and how cross entropy, perplexity and bits per byte measure the prediction."
sidebar:
  order: 1
  label: "Language Modeling"
lastUpdated: 2026-11-18
---

Lily opens the attic door and spots a cat on the windowsill. A model telling the story could follow “the cat” with “sat” or “jumped”, and whichever word it adds will change what makes sense next. If it chooses “jumped”, “down” becomes a natural next word.

A **language model** assigns probabilities to possible continuations one piece at a time, a process called **next-token prediction**. The pieces, called **tokens**, can be whole words, parts of words, spaces, or punctuation. The [story model](/tutorials/llm-fundamentals/#story-model) has 4,096 entries in its **vocabulary**, so each prediction gives a probability for every one of those entries. During generation the model uses that distribution to choose what to append. When we score existing text, such as a story in the training set, we read the probability the model assigned to the token that actually appeared.

![Next token chain](/images/llm-fundamentals/15a-next-token-chain.png)

Each panel shows a few of the possible continuations at one position. To score “the cat sat”, we pick out the probability of “the” in the first panel, “cat” in the second, and “sat” in the third.

## Text as a token sequence

A **tokenizer** turns text into a sequence of integer IDs the model can process. It splits the text into pieces and looks up each piece in the vocabulary. Smaller pieces let it represent words that have no ID of their own. The [Tokenization](/tutorials/llm-fundamentals/tokenization-and-embeddings/) chapter explains how those pieces are chosen.

Suppose a simple tokenizer represents “the cat sat” with the IDs `[12, 47, 83]`. The number 47 identifies “cat” wherever that token appears, while its place in this list tells us that “cat” is the second token in this sentence.

To predict “sat”, the model has the first two IDs, `[12, 47]`. Those earlier tokens make up the **prefix** for that prediction. We write a sequence of $T$ tokens as $x_1, \ldots, x_T$, so at position $t$ the model predicts $x_t$ from the prefix $x_{<t}$.

The prefix changes at every step. At the start, it is empty. When the model scores “cat”, it has already seen “the”, and when it scores “sat”, it has seen “the cat”. Each of those probabilities answers a slightly different question about the same sentence.

## The chain rule of probability

A model might think “sat” is a likely continuation of “the cat” while rarely starting a sentence with “the cat” in the first place. The probability of all three words has to account for the choices that got the model to that point.

In the opening figure, “the” gets a 20% chance at the start of the text. Only 10% of the probability from that branch goes to “cat”, so the chance of starting with both words is $0.2 \times 0.1 = 0.02$. From “the cat”, the chance of continuing with “sat” is 30%, which leaves $0.02 \times 0.3 = 0.006$, or a 0.6% chance of starting with “the cat sat”.

The 10% is a **conditional probability** because it assumes “the” has already appeared, and we write it as $P(\text{cat} \mid \text{the})$. The **chain rule** multiplies these probabilities in order, using everything that came before each token as its condition.

$$
P_\theta(x_1, \ldots, x_T) = \prod_{t=1}^{T} P_\theta(x_t \mid x_{<t}).
$$

The subscript $\theta$ denotes the model’s parameters, and $\prod$ multiplies one factor for every position from 1 through $T$. The first factor uses an empty prefix, so it measures how likely the token is at the start of the text.

Generation uses the same growing prefix. After the model chooses a token, it appends it and predicts again with that token in view. This is called **autoregressive** generation.

A long sequence can contain hundreds of factors, and their product can get surprisingly small. If each observed token in a 1,000-token document gets probability 0.1, the product is $10^{-1000}$, which rounds to zero in 32-bit floating point. Taking the natural logarithm gives us the **log probability**, which replaces the product with a sum that we can store.

$$
\ln(0.2 \times 0.1 \times 0.3) = \ln 0.2 + \ln 0.1 + \ln 0.3 \approx -5.116.
$$

Exponentiating the sum recovers about 0.006. For scoring or training, we can work with the sum directly, and the loss in the [cross entropy loss](#cross-entropy-loss) section uses these same log probabilities with their signs flipped.

## N-gram models

An **n-gram** is a run of $n$ consecutive tokens. An n-gram model estimates what comes next from how often these runs occur in the training text. That collection of text is called the **corpus**.

The longer the prefix, the harder it is to find examples that match it exactly. An n-gram model makes the **Markov assumption**, which here means estimating the probability from just the last $n-1$ tokens. A trigram model can distinguish “the cat” from “a cat”, while a bigram model uses “cat” as its context in both cases.

Consider the small corpus “the cat sat . the dog sat . the cat ran .”, with each period counted as a token. “Cat” follows “the” in two of its three occurrences, so a bigram model gives $P(\text{cat} \mid \text{the}) = 2/3$. In general, $c(x_{t-1}, x_t)$ counts how often the pair occurs, and $c(x_{t-1})$ counts all the times that context was followed by a token.

$$
P(x_t \mid x_{t-1}) = \frac{c(x_{t-1}, x_t)}{c(x_{t-1})}.
$$

The same rule gives “sat” probability 1 after “dog”, because that was the only continuation the model saw. It gives “ran” zero, which is enough to make the chain-rule probability of “the dog ran” zero too. This is the **zero count problem**, which arises whenever a plausible sequence is missing from the training counts.

Longer contexts give the model more information, but they also make exact matches harder to find in the training text.

### Smoothing

Raw counts give a continuation zero probability if it never followed the current context in the training data. **Smoothing** gives these unseen continuations some probability.

**Add-one smoothing** gives one extra count to each of the six tokens in the vocabulary. After “dog”, that raises the total from 1 to 7. Each token’s new probability is its updated count divided by 7. “Sat” gets $2/7$ because it has one observed count and one added count. “Ran” gets $1/7$ from its added count alone.

![After “dog”, the raw counts give “sat” 100% probability and the other five tokens 0%. Adding one count per token raises the total count from 1 to 7. “Sat” gets 2/7 and each of “ran”, “the”, “cat”, “dog”, and the period gets 1/7.](/images/llm-fundamentals/15a-add-one-smoothing.png)

Adding a full count to every possibility can overwhelm the data when the vocabulary is large. The project’s bigram baseline adds 0.1 per possible continuation, giving observed counts more weight. Another approach discounts the observed pairs a little and uses shorter contexts to distribute the freed probability. Both approaches can give “ran” some probability after “dog”, but the counts themselves offer no way to learn that “cat” and “dog” often behave similarly.

## Neural language models

A **neural language model** learns representations of tokens and the contexts they appear in. What the model learns about one token can help it make predictions from another that appears in similar contexts. That lets it make informed predictions about combinations it never saw during training. Transformer language models can also draw on information from farther back in the prefix.

The model looks up each token’s **embedding**, a learned vector of numbers. The [story model](/tutorials/llm-fundamentals/#story-model) gives each of its 4,096 tokens a 384-dimensional vector. The lookup gives “cat” the same starting vector in “the cat” and “the black cat”, while the surrounding words can change how the model interprets it. The [Tokenization](/tutorials/llm-fundamentals/tokenization-and-embeddings/) chapter explains how these vectors are stored and learned.

The [Transformer blocks](/tutorials/llm-fundamentals/the-transformer-block-and-model-anatomy/) combine information from the prefix into a **hidden state** $h_t$ for the prediction position. The **output head** compares that state with an output vector $u_w$ for each possible token $w$. The comparison is a dot product, $h_t \cdot u_w$, which multiplies matching coordinates and adds them. The story model reuses the embedding vectors as its output vectors, a design called [weight tying](/tutorials/llm-fundamentals/tokenization-and-embeddings/#weight-tying).

![Neural language model stages](/images/llm-fundamentals/15a-neural-language-model-stages.png)

To see how sharing information works, imagine a two-dimensional model whose hidden state is just the embedding of the last token. Give “cat” the vector $(1.0, 0.5)$, “dog” the vector $(0.9, 0.6)$, and “ran” the output vector $(2, 0)$. The score for “ran” after “cat” is $1.0 \times 2 = 2.0$, while after “dog” it is $0.9 \times 2 = 1.8$.

![Embedding scores](/images/llm-fundamentals/15a-embedding-scores.png)

If one training update moves the first coordinate of the “ran” vector from 2.0 to 2.1 and holds the other vectors fixed, the score rises by 0.10 after “cat” and by 0.09 after “dog”. One update affects both predictions through the shared output vector. To turn the resulting scores into next-token probabilities, the model has to account for all the scores together.

## Logits and softmax

After “the cat”, the output head might score “sat” at 2.0, “ran” at 1.0, “is” at 0.5, and “the” at −1.0. These raw scores are called **logits**. They tell us that the model ranks “sat” highest, but we still need a distribution to tell us how much more likely it is than the other choices.

**Softmax** first gives each token a positive weight by exponentiating its logit, then divides each weight by the total. The weight for “sat” is $e^2 \approx 7.39$, which is about 0.609 of the total weight of 12.12.

$$
P_\theta(w \mid x_{<t}) = \frac{e^{z_w}}{\sum_{v} e^{z_v}}.
$$

The numerator is the weight for candidate $w$, and the denominator counts the weight of every candidate $v$, so the shares add up to 1. In this example, “ran” gets about 0.224, “is” gets 0.136, and “the” gets 0.030.

Exponentiating very large logits can overflow and make softmax numerically unstable. The usual fix is to subtract the largest logit from every logit before exponentiating. For logits of 1,000 and 1,001, the shifted values are −1 and 0, so both exponentials are at most 1. The shift scales every weight by the same factor, which cancels when we divide by their sum. The larger logit still gets a probability of about 0.731.

This shift makes [stable softmax](/tutorials/llm-deep-dive/efficient-attention-mechanisms/#online-softmax) possible. With $m$ as the largest logit, the model can compute the log probability directly, without rounding a very small probability before taking its log.

$$
\ln P_\theta(w \mid x_{<t}) = (z_w-m) - \ln\sum_v e^{z_v-m}.
$$

The last term is the **log-sum-exp** of the shifted scores. It is the same normalization as dividing by the total weight, written as a subtraction in log space. The training loss uses this form.

When the model generates text, **temperature** $\tau$ changes how heavily it favors the highest score. Replacing $z_w$ with $z_w/\tau$ makes the gaps larger at low temperatures and smaller at high ones. With temperature 0.5, “sat” rises from about 0.609 to 0.842. With temperature 2, its probability falls to 0.434 and the other candidates have more chance of being picked.

<details class="cm-steps not-content" data-anim="15a-softmax-walkthrough">
<summary class="cm-steps-head"><span class="cm-steps-title"><span class="cm-steps-label">Step-by-step example:</span> Softmax</span><span class="cm-collapse-hint" aria-hidden="true"></span><span class="cm-steps-sub">After the context &quot;the cat&quot;, four candidate tokens get raw scores of 2.0, 1.0, 0.5 and −1.0. The example turns them into probabilities that sum to one, then shows how temperature sharpens or flattens them.</span></summary>
<ol class="cm-steps-list">
<li class="cm-step"><p class="cm-step-text"><span class="cm-step-num">Step 1</span>The network ends with one raw score per vocabulary token, called a logit. Logits can be any real number, negative included, and they do not sum to anything in particular.</p><img src="/images/steps/15a-softmax-walkthrough/step-01.png" alt="Step 1 of 7: The network ends with one raw score per vocabulary token, called a logit. Logits can be any real number, negative included, and they do not sum to anything in particular." loading="lazy" decoding="async"></li>
<li class="cm-step"><p class="cm-step-text"><span class="cm-step-num">Step 2</span>At temperature 1 the logits pass through unchanged. Temperature is a number that divides every logit before the next step.</p><img src="/images/steps/15a-softmax-walkthrough/step-02.png" alt="Step 2 of 7: At temperature 1 the logits pass through unchanged. Temperature is a number that divides every logit before the next step." loading="lazy" decoding="async"></li>
<li class="cm-step"><p class="cm-step-text"><span class="cm-step-num">Step 3</span>Exponentiating each logit turns 2.0 into 7.389 and -1.0 into 0.368. Every result is positive and the larger logit stays larger.</p><img src="/images/steps/15a-softmax-walkthrough/step-03.png" alt="Step 3 of 7: Exponentiating each logit turns 2.0 into 7.389 and -1.0 into 0.368. Every result is positive and the larger logit stays larger." loading="lazy" decoding="async"></li>
<li class="cm-step"><p class="cm-step-text"><span class="cm-step-num">Step 4</span>Dividing each exponential by their sum, 12.124, gives the probabilities 0.609, 0.224, 0.136 and 0.030, and they sum to one.</p><img src="/images/steps/15a-softmax-walkthrough/step-04.png" alt="Step 4 of 7: Dividing each exponential by their sum, 12.124, gives the probabilities 0.609, 0.224, 0.136 and 0.030, and they sum to one." loading="lazy" decoding="async"></li>
<li class="cm-step"><p class="cm-step-text"><span class="cm-step-num">Step 5</span>Subtracting the largest logit from all four first gives the same probabilities, because the shared factor cancels in the division. Real code does this so e to a large logit never overflows.</p><img src="/images/steps/15a-softmax-walkthrough/step-05.png" alt="Step 5 of 7: Subtracting the largest logit from all four first gives the same probabilities, because the shared factor cancels in the division. Real code does this so e to a large logit never overflows." loading="lazy" decoding="async"></li>
<li class="cm-step"><p class="cm-step-text"><span class="cm-step-num">Step 6</span>Temperature 0.5 doubles every logit gap before exponentiating. &quot;sat&quot; rises to 0.842 and &quot;the&quot; falls to 0.002, so sampling becomes nearly deterministic.</p><img src="/images/steps/15a-softmax-walkthrough/step-06.png" alt="Step 6 of 7: Temperature 0.5 doubles every logit gap before exponentiating. &quot;sat&quot; rises to 0.842 and &quot;the&quot; falls to 0.002, so sampling becomes nearly deterministic." loading="lazy" decoding="async"></li>
<li class="cm-step"><p class="cm-step-text"><span class="cm-step-num">Step 7</span>Temperature 2 halves every gap. The distribution flattens to 0.434, 0.263, 0.205 and 0.097, so sampling picks the unlikely tokens more often.</p><img src="/images/steps/15a-softmax-walkthrough/step-07.png" alt="Step 7 of 7: Temperature 2 halves every gap. The distribution flattens to 0.434, 0.263, 0.205 and 0.097, so sampling picks the unlikely tokens more often." loading="lazy" decoding="async"></li>
</ol>
<div class="cm-steps-notes">
<p>A decoder that always picks the highest-scoring token would still choose “sat” at both settings. Temperature changes how often each token is chosen when <strong>sampling</strong>, which draws a token at random from the distribution. The training loss uses temperature 1, and the <a href="/tutorials/llm-fundamentals/inference-and-decoding/">Inference</a> chapter explores these settings during generation.</p>
</div>
</details>

## Cross entropy loss

A **loss function** turns the model’s predictions into a number that training tries to lower. For next-token prediction, **negative log likelihood**, or NLL, takes the negative log of the probability assigned to the observed token. The less probability the model gives that token, the larger the penalty.

The next-token loss averages those penalties over the $T$ tokens being scored.

$$
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \ln P_\theta(x_t \mid x_{<t}).
$$

We already calculated the sum of the log probabilities for “the cat sat” in the [chain rule of probability](#the-chain-rule-of-probability) section. Flipping the sign gives a total loss of about 5.116 **nats**, the unit we get from using natural logs, and dividing by three gives 1.705 nats per token. “Cat” contributes the most to that total because it received the least probability.

After “the cat”, the model’s probabilities for all the possible next tokens make up its **predicted distribution**. The **target distribution** is the set of probabilities training asks the model to match. When the text says “the cat sat”, the target is **one-hot**. It gives “sat” probability 1 and every other token probability 0.

In token-level [distillation](/tutorials/llm-fundamentals/post-training-from-base-model-to-assistant/#distillation), a separate model called the **teacher** predicts what comes after the same prefix. The **teacher distribution** is that model’s set of next-token probabilities. It might give “sat” 0.7 and “ran” 0.2, with 0.1 left for other tokens. The model being trained uses those probabilities as its targets so it can learn from several possible continuations.

At position $t$, **cross entropy** multiplies the negative log of the model’s prediction for each token $w$ by that token’s target probability $y_{t,w}$. It sums those terms and averages the result over the $T$ positions.

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{T}\sum_{t=1}^{T}\sum_w y_{t,w}\ln P_\theta(w \mid x_{<t}).
$$

With a one-hot target, only the term for “sat” survives, giving the same loss as NLL. With the teacher’s probabilities as the target, the loss includes 0.7 times the penalty for “sat”, 0.2 times the penalty for “ran”, and the contributions from the other tokens.

The [maximum likelihood](#maximum-likelihood) section connects this per-token loss to the probability of the whole training text and explains what minimizing it does to the model’s next-token distribution.

## Perplexity

**Perplexity** puts next-token loss on a scale that makes it easier to compare models on held-out text. A perplexity of 6 means the observed tokens got the same average loss as uniform guesses among six choices per position. On the same text, a lower score means the model assigned a higher probability to the observed token sequence.

A uniform guess among six tokens gives each one probability $1/6$, for a loss of $\ln 6$ nats. Exponentiating that loss gives the six choices back. Perplexity uses the same conversion on the model’s average negative log likelihood $\mathcal{L}$ in nats.

$$
\text{PP} = e^{\mathcal{L}} = \left(\prod_{t=1}^{T} \frac{1}{P_\theta(x_t \mid x_{<t})}\right)^{1/T}.
$$

For “the cat sat”, the model gave “the”, “cat” and “sat” probabilities of 0.2, 0.1 and 0.3 as it scored them in order. Multiplying those probabilities gives 0.006 for the whole sequence. Now imagine a model that gives the observed token the same probability $q$ at every position. To keep the sequence probability at 0.006, three copies of $q$ must multiply to 0.006.

$$
q \times q \times q = 0.006, \qquad q = \sqrt[3]{0.006} \approx 0.182.
$$

A uniform guess among six choices gives each choice probability $1/6$. Here the probability is $q$, so the equivalent number of choices is $1/q \approx 5.50$. That is the perplexity. Exponentiating the average loss of 1.705 nats per token gives the same answer.

<details class="cm-steps not-content" data-anim="15a-perplexity-walkthrough">
<summary class="cm-steps-head"><span class="cm-steps-title"><span class="cm-steps-label">Step-by-step example:</span> Perplexity</span><span class="cm-collapse-hint" aria-hidden="true"></span><span class="cm-steps-sub">A model scores the sentence &quot;the cat sat&quot;, giving the token that really came next probability 0.2, then 0.1, then 0.3. The example turns those three probabilities into an average loss in nats and then into a perplexity.</span></summary>
<ol class="cm-steps-list">
<li class="cm-step"><p class="cm-step-text"><span class="cm-step-num">Step 1</span>At the start of text the model gave &quot;the&quot; probability 0.2. The loss is minus the natural log, -ln 0.2 = 1.609 nats.</p><img src="/images/steps/15a-perplexity-walkthrough/step-02.png" alt="Step 1 of 5: At the start of text the model gave &quot;the&quot; probability 0.2. The loss is minus the natural log, -ln 0.2 = 1.609 nats." loading="lazy" decoding="async"></li>
<li class="cm-step"><p class="cm-step-text"><span class="cm-step-num">Step 2</span>After &quot;the&quot; it gave &quot;cat&quot; only 0.1, so this position costs 2.303 nats, the largest surprise in the sentence.</p><img src="/images/steps/15a-perplexity-walkthrough/step-03.png" alt="Step 2 of 5: After &quot;the&quot; it gave &quot;cat&quot; only 0.1, so this position costs 2.303 nats, the largest surprise in the sentence." loading="lazy" decoding="async"></li>
<li class="cm-step"><p class="cm-step-text"><span class="cm-step-num">Step 3</span>After &quot;the cat&quot; it gave &quot;sat&quot; 0.3, costing 1.204 nats. The total is 5.116 nats and the sentence probability is 0.2 × 0.1 × 0.3 = 0.006.</p><img src="/images/steps/15a-perplexity-walkthrough/step-04.png" alt="Step 3 of 5: After &quot;the cat&quot; it gave &quot;sat&quot; 0.3, costing 1.204 nats. The total is 5.116 nats and the sentence probability is 0.2 × 0.1 × 0.3 = 0.006." loading="lazy" decoding="async"></li>
<li class="cm-step"><p class="cm-step-text"><span class="cm-step-num">Step 4</span>The loss averages 1.705 nats over the 3 tokens, and exponentiating that average gives the perplexity. Perplexity 5.50 means the model was as uncertain as a fair choice among about 5.5 tokens at each step.</p><img src="/images/steps/15a-perplexity-walkthrough/step-05.png" alt="Step 4 of 5: The loss averages 1.705 nats over the 3 tokens, and exponentiating that average gives the perplexity. Perplexity 5.50 means the model was as uncertain as a fair choice among about 5.5 tokens at each step." loading="lazy" decoding="async"></li>
<li class="cm-step"><p class="cm-step-text"><span class="cm-step-num">Step 5</span>A model that spreads probability evenly over the story model's 4,096 tokens pays ln 4,096 = 8.32 nats on every token, so its perplexity is the vocabulary size. The untrained story model scored 8.08 nats, a perplexity of 3,219, close to this number.</p><img src="/images/steps/15a-perplexity-walkthrough/step-06.png" alt="Step 5 of 5: A model that spreads probability evenly over the story model's 4,096 tokens pays ln 4,096 = 8.32 nats on every token, so its perplexity is the vocabulary size. The untrained story model scored 8.08 nats, a perplexity of 3,219, close to this number." loading="lazy" decoding="async"></li>
</ol>
<div class="cm-steps-notes">
<p>The running score rises from 5.0 after “the” to 7.07 after the unexpected “cat”, then settles at 5.50 after “sat”. It swings sharply because we are averaging only three predictions. A larger set of text held out from training gives a better read on how the model predicts new text.</p>
</div>
</details>

That shared scale is useful for comparing a trained model with a baseline or checking how predictions improve across checkpoints. For example, on the same held-out TinyStories text, the bigram baseline has a perplexity of 23.8 and the trained story model has a perplexity of 2.86. That is roughly the difference between uniform guesses among 24 choices and among 3 choices per token. Both models use the same tokenizer, so they are scoring the same sequence of tokens. [Evaluation and Safety](/tutorials/llm-fundamentals/evaluation-reliability-and-safety/#intrinsic-and-task-evaluation) explains how to check task quality separately from fit to text.

<details class="cm-collapse" id="bits-per-byte-across-tokenizers">
<summary><span class="cm-collapse-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg></span><span class="cm-collapse-text"><span class="cm-collapse-kicker">Deep dive</span><span class="cm-collapse-title">Bits per byte across tokenizers</span></span><span class="cm-collapse-hint" aria-hidden="true"></span></summary>

Perplexity is based on average loss per token. If one tokenizer splits a sentence into six tokens and another splits it into nine, the two scores average over different units. Some of the extra tokens may be easy to predict, like the last piece of a word. That can bring perplexity down even when the total loss is higher. **Bits per byte**, or BPB, divides the total loss in bits by the byte count of the original text. That byte count is the same for both models.

Base-2 logs express prediction loss in bits. For example, if the model gives “sat” probability 0.5, the token adds $-\log_2 0.5 = 1$ bit to the ideal compressed length.

$$
\text{BPB} = \frac{\sum_{t=1}^{T}-\log_2 P_\theta(x_t\mid x_{<t})}{B} = \frac{T\mathcal{L}}{B\ln 2}.
$$

Here $B$ is the UTF-8 byte count, and $T\mathcal{L}$ is the total loss in nats over the tokens that encode that text. Dividing by $\ln 2$ converts the total to bits.

In the figure, Model A spends 12.0 nats on six tokens and Model B spends 12.6 on nine. Model B’s extra tokens bring its perplexity down to 4.06 versus 7.39 for Model A. Both models are scoring the same 22 bytes, so Model A’s lower total gives it the better bits-per-byte score, 0.787 versus 0.826.

![Bits per byte](/images/llm-fundamentals/15a-bits-per-byte.png)

</details>

## Maximum likelihood

Suppose the training text says “the cat sat”. A model that gives “sat” a 60% chance after “the cat” does better on that token than one that gives it 30%. **Maximum likelihood** takes the whole training text into account. The goal is to give the text we actually saw the highest possible probability.

In the example from the [cross entropy loss](#cross-entropy-loss) section, the model gave “the cat sat” a probability of 0.006. The loss takes the negative log and divides by three because the sentence has three tokens. That gives $-\ln(0.006)/3 \approx 1.705$ nats per token. A higher probability gives a lower loss, so minimizing this one-hot cross entropy is another way to maximize the probability of the training text.

### Data entropy and KL divergence

Imagine ten examples of “the cat”. Five continue with “sat” and five with “ran”. Suppose these are the only two continuations and both are equally likely. Those next-token probabilities make up the **data distribution** $P_{\text{data}}$.

A model that predicts 50–50 gets the same loss on either token, $-\ln 0.5 \approx 0.693$ nats. A model that gives “sat” a 90% chance gets a small loss of 0.105 when “sat” appears. It gives “ran” only a 10% chance and gets a loss of 2.303 when “ran” appears. Since both tokens appear equally often, each loss gets half the weight.

$$
0.5(-\ln 0.9) + 0.5(-\ln 0.1) \approx 1.204\text{ nats}.
$$

The 50–50 model gets the odds right, and there is still uncertainty about which token will come next. Its loss of 0.693 nats is the **entropy** of the data. The 90–10 model pays an extra 0.511 nats because its predictions differ from those odds. That extra loss is the **Kullback-Leibler divergence**, or KL divergence, from the data to the model. KL measures how far off the model’s odds are by looking at how much extra loss they cause on average.

Let $P_\theta$ be the model’s probabilities. In the formula, $H(P_{\text{data}}, P_\theta)$ is the model’s average loss, and $H(P_{\text{data}})$ is the loss it would get by matching the data exactly.

$$
H(P_{\text{data}}, P_\theta) = H(P_{\text{data}}) + D_{\text{KL}}(P_{\text{data}} \,\Vert\, P_\theta).
$$

The KL formula adds up the difference in loss for each possible next token $w$. The log ratio gives that difference for one token, and the data probability tells us how often to count it.

$$
D_{\text{KL}}(P_{\text{data}} \,\Vert\, P_\theta) = \sum_w P_{\text{data}}(w) \ln \frac{P_{\text{data}}(w)}{P_\theta(w)}.
$$

Training can shrink the KL gap by improving the model’s probabilities. That is how learning from one token at a time can teach the model the odds of all the possible continuations. Cross entropy already includes the gap, so adding KL against the same target counts it twice.

The 50–50 target describes what follows “the cat” across many examples. The same formula also works for the one-hot target at a single position in the training text.

Suppose one training example says “the cat sat”. The one-hot target gives “sat” 100% and “ran” 0%. There is no uncertainty in that target, so its entropy is $-\ln 1 = 0$. If the model predicts 90% for “sat”, cross entropy only counts $-\ln 0.9$. That is also the NLL on “sat”. Since cross entropy is entropy plus KL, subtracting the zero entropy leaves the same number for KL.

$$
\text{Cross entropy} = \text{NLL} = -\ln 0.9 \approx 0.105\text{ nats}.
$$

$$
\text{KL from the one-hot target} = \ln(1/0.9) = -\ln 0.9.
$$

Now suppose a teacher predicts 70% “sat” and 30% “ran” for the same prefix. The model still predicts 90% “sat” and 10% “ran”. The NLL on the observed “sat” still uses just the model’s 90%. KL from the teacher to the model compares both pairs of probabilities. Matching the teacher would raise the model’s probability for “ran” from 10% toward 30%.

A distillation loss can use both signals. The observed token tells the model what actually came next. The teacher’s probabilities also tell it how plausible the other continuations were according to the teacher.

### Mode covering and mode seeking

A **mode** is a peak in a probability distribution. In the figure, the data has two equally common groups. One is near −2 and the other is near +2, with very few values near 0. Let’s assume the model can fit just one bell curve. It can use a wide curve that reaches both groups or a narrow curve that fits one group closely.

Maximum likelihood minimizes **forward KL**, $D_{\text{KL}}(P_{\text{data}} \Vert P_\theta)$. It gives more weight to outcomes that appear often in the data. A narrow curve near +2 would give very little probability to values near −2, even though they make up about half the examples. Those examples would contribute a large loss. The wide curve centered at 0 gives both groups some probability and also assigns probability to the gap between them. This is **mode covering**. Forward KL fits a training objective that rewards the model for predicting the examples it sees, including examples from every common group in the data.

**Reverse KL**, $D_{\text{KL}}(P_\theta \Vert P_{\text{data}})$, gives more weight to outcomes the model is likely to produce. A wide curve often produces values near 0, where the data rarely goes, so those values contribute a large penalty. The narrow curve shown near +2 produces mostly values from one real group. It gives very little probability to the gap or the other group. Since the model rarely produces values near −2, missing that group carries little direct weight in reverse KL. This is **mode seeking**. Reverse KL is useful when the model needs to generate plausible examples and it’s acceptable to cover only some of the patterns in the data.

![Mode covering](/images/llm-fundamentals/15a-mode-covering.png)

The one-curve limit forces a choice about where to put the probability. If the model matched the entire data distribution, KL would be zero in both directions.

## Teacher forcing and exposure bias

Suppose the training text says “the cat sat”. After “the”, the model’s top guess is “dog”, and it gives “cat” only a 10% chance. The loss for that prediction is $-\ln 0.10 \approx 2.30$ nats. For the next prediction, the model sees “the cat” from the training text and tries to predict “sat”. Giving the model the actual earlier tokens at each position is called **teacher forcing**.

The input tokens are `["the", "cat"]` and the targets are `["cat", "sat"]`. The target list starts one token later because each position predicts the token that follows it. In code, the input drops the last token of the sentence and the targets drop the first.

Teacher forcing is useful when we have example text to learn from. In pretraining that text comes from the corpus. In [supervised fine-tuning](/tutorials/llm-fundamentals/post-training-from-base-model-to-assistant/#supervised-fine-tuning), it can be a reference answer. Because all the input tokens are known up front, the model can score every position in one pass. The [causal mask](/tutorials/llm-fundamentals/attention-and-positional-information/#the-causal-mask) lets each position read its prefix while hiding the tokens that come later.

![Teacher forcing and generation](/images/llm-fundamentals/15a-teacher-forcing-and-generation.png)

### Exposure bias

In the figure, the model starts with “the” and samples “dog”. It now has “the dog” as the prefix for its next prediction. In autoregressive generation, the model repeats that step for each new token it writes. In this run, the first choice leads to “the dog ran to the park”.

**Exposure bias** is the gap between learning from prefixes in the training text and generating from prefixes the model wrote itself. During generation, an early mistake can become part of the context for everything that follows. For example, if the model writes down the wrong subtotal, it can do all the remaining arithmetic correctly for that subtotal and still reach the wrong answer.

Teacher-forced loss measures how well the model predicts the next token when it reads the earlier tokens from the reference text. When the model writes the whole answer from the prompt, we can see whether it catches and corrects a mistake like the wrong subtotal. Some [reinforcement learning methods](/tutorials/llm-fundamentals/post-training-from-base-model-to-assistant/#rlhf) use this setup for training too. The model writes an answer, and the score it receives on the finished answer guides the update.

