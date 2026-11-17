---
title: "Language Modeling"
description: "How the chain rule turns a sentence into a product of next-token probabilities, why counting word sequences breaks down, and how a neural network shares statistical strength across contexts."
sidebar:
  order: 1
  label: "Language Modeling"
lastUpdated: 2026-11-13
---

Your phone's keyboard guesses the next word from the words already typed. A chatbot writing an essay does the same thing thousands of times in a row. Each guess is a set of odds over every word that could come next, and whichever word gets picked becomes part of the text that the next guess reads.

The program behind both is a **language model**, which gives a probability to every possible continuation of a piece of text. It works in **tokens**, units of text that are usually a whole word or a piece of one, and it can only produce tokens from its **vocabulary**, a fixed list that holds 32,000 entries in the reference 7B model used across this series. Turning the text so far into one probability for each vocabulary entry is called **next token prediction**, and writing, scoring and training all reduce to it. Writing repeats the prediction and appends one chosen token each time. Scoring a sentence multiplies the probabilities the model gave to the tokens that really appeared, and training adjusts the model so those probabilities rise. The figure shows the three predictions behind the short sentence "the cat sat".

![Next token chain](/images/llm-fundamentals/15a-next-token-chain.png)

Each panel is a full probability distribution over the vocabulary, but scoring and training read only one bar per panel, the one for the token that actually followed. Everything later in the chapter, from counting to perplexity, is a different way of producing those panels or of summarizing the bars that were read.

One question runs through every section below: how to get good next-token probabilities, and how to know they are good.

Producing the probabilities comes first. Text becomes a sequence of tokens, and the chain rule splits a sentence's probability into one prediction per token. Counting is the simplest way to make each prediction, and it breaks as soon as a context has never been seen before. A neural network replaces it by scoring every token and turning the scores into probabilities with softmax.

Then comes the measuring stick. Cross entropy is the loss that training pushes down. Perplexity turns that loss into a number that is easy to read, and bits per byte keeps the comparison fair when two models use different tokenizers. The closing sections ask what this training really achieves, which is matching the distribution of real text, and where that falls short once the model has to write on its own.

## Text as a token sequence

A computer stores text as a list of numbers, and a language model needs a fixed list of units to predict. One could use letters, which gives a tiny vocabulary but very long sequences, or whole words, which gives short sequences but cannot spell a name it has never seen. A **tokenizer** is the program that cuts raw text into tokens and maps each one to an integer ID between 0 and $V-1$. Modern tokenizers cut common words into one token and rare words into several pieces, so "the cat sat" becomes three IDs, one per word, while "photosynthesis" is usually cut up. The GPT-2 tokenizer splits it into "photos" and "ynthesis", and the tokenizer built for BERT, an earlier model for understanding text, splits it into "photos", "yn", and "thesis", which shows that the cuts follow how often letter sequences appear in the training text rather than where the meaningful parts "photo" and "synthesis" meet. A tokenizer with a larger or more science-heavy vocabulary, such as DeepSeek-V3's, keeps the whole word as one token. How the pieces are chosen is the subject of [Tokenization](/tutorials/llm-fundamentals/tokenization-and-embeddings/), and this chapter treats the cut as already made. After tokenization a document is a sequence $x_1, x_2, \ldots, x_T$, where $x_t$ is the token at position $t$ and $T$ is the length. The **prefix** $x_{<t}$ is everything before position $t$, and it is all the model is allowed to see when predicting $x_t$. Under the hood text is also stored as bytes, the 8-bit units a file is made of, with one byte per ordinary English letter in the common UTF-8 encoding. We will come back to bytes in the deep dive on bits per byte in the [perplexity](#perplexity) section, since every tokenizer agrees on how many bytes a text has, which makes them a fair unit for comparing models. The reference 7B model reads up to 4,096 tokens at once and chooses from 32,000 vocabulary entries at every position. A 4,096-token document therefore poses 4,095 separate prediction problems, each with 32,000 possible answers, and pretraining on 2 trillion tokens poses about 2 trillion of them.

## The chain rule of probability

Guessing a whole sentence at once is hopeless, since even short sentences are almost all unique, but guessing one word given the words before it is manageable. A **conditional probability** is the probability of one event once you already know another has happened, written $P(A \mid B)$ and read "the probability of A given B". The **chain rule** of probability says the probability of several events together equals the probability of the first, times the probability of the second given the first, and so on. For text it splits one impossible question into $T$ manageable ones:

$$
P(x_1, \ldots, x_T) = \prod_{t=1}^{T} P_\theta(x_t \mid x_{<t}).
$$

Here $\prod$ means multiply the factors for $t = 1$ through $T$, $x_t$ is the token at position $t$, $x_{<t}$ is the prefix before it, and $P_\theta$ is the model with parameters $\theta$. The first factor conditions on an empty prefix, meaning the start of the text. The equation is exact rather than an approximation, so a model that gets every next-token probability right gets every sentence probability right. A model that generates this way, one token at a time with each new token conditioned on all earlier ones, is called **autoregressive**.

In the opening figure, the model gave "the" probability 0.2 at the start, "cat" probability 0.1 after "the", and "sat" probability 0.3 after "the cat". The sentence probability is $0.2 \times 0.1 \times 0.3 = 0.006$. Multiplying many small numbers creates a practical problem. If every token of a 1,000-token document had probability 0.1, the document's probability would be $10^{-1000}$, far below the smallest positive number a 32-bit float, the usual format a computer uses for a decimal number, can hold, which is about $10^{-45}$, so the computer would store exactly zero and every long document would look equally impossible. Working with logarithms avoids this, because the log of a tiny probability is an ordinary negative number, such as $\ln 10^{-1000} \approx -2{,}303$. The **log probability**, the natural logarithm of a probability, turns the product into a sum:

$$
\ln 0.2 + \ln 0.1 + \ln 0.3 = -1.609 - 2.303 - 1.204 = -5.116.
$$

Taking $e^{-5.116} = 0.006$ recovers the product. Every practical system adds log probabilities, and the loss in the [cross entropy loss](#cross-entropy-loss) section is the same sum with its sign flipped.

## The simplest language model: counting

The simplest way to guess the word after "the cat" is to look through a lot of text and count what usually comes next. If "the cat" shows up 100 times and is followed by "sat" 30 of those times, the guess for "sat" is 30 out of 100, or 0.3. The text you count in is called a **corpus**, the collection of text a model learns from. The short runs of words you count are called **n-grams**, where $n$ is how many tokens are in the run: "the cat" is a 2-gram, or bigram, and "the cat sat" is a 3-gram, or trigram. Counting only works for short histories, because almost any long stretch of text, such as a whole paragraph, has never appeared before and has a count of zero. So a counting model makes the **Markov assumption**: it remembers only the last $n-1$ tokens and forgets everything earlier. The "the cat" example remembers two words, which makes it a trigram model, while a bigram model remembers just one. Speech recognizers in the 1990s predicted the next word this way, from counts alone. Turning counts into a probability means dividing how often the pair appears by how often its first word appears. This rule is called the **maximum likelihood estimate**, because it picks the probabilities that make the counted text as likely as possible. For a bigram model it is:

$$
P(x_t \mid x_{t-1}) = \frac{c(x_{t-1}\, x_t)}{c(x_{t-1})}.
$$

Here $c(x_{t-1}\, x_t)$ counts how often the pair occurs in the corpus and $c(x_{t-1})$ counts how often the first token occurs. A useful toy case is the twelve-token corpus "the cat sat . the dog sat . the cat ran .", where the period counts as a token. "the" occurs 3 times and "the cat" twice, so $P(\text{cat} \mid \text{the}) = 2/3$. "dog" occurs once and is followed by "sat", so $P(\text{sat} \mid \text{dog}) = 1$ and $P(\text{ran} \mid \text{dog}) = 0$. That zero is the central failure of counting. The **zero count problem** is that any sequence missing from the corpus gets probability exactly zero, and by the chain rule one zero factor makes the whole sentence probability zero and its log probability minus infinity. "the dog ran" is a perfectly ordinary sentence, but this model calls it impossible because "dog ran" never appeared. **Sparsity**, the fact that most possible n-grams never appear in any finite corpus, makes the zero count problem the normal case once $n$ reaches three. The figure measures this on real text, about 1.9 million words of the author's study notes, with a tenth of the files held back as a test.

![Ngram sparsity](/images/llm-fundamentals/15a-ngram-sparsity.png)

On that text 76% of held-back three-word sequences never occur in the 1.7 million training words, and about eleven times more training text barely helps at four words and beyond. At the scale of the reference 7B model the arithmetic is starker. With $V = 32{,}000$ there are $32{,}000^3 \approx 3.3 \times 10^{13}$ possible trigrams, and a 2-trillion-token corpus contains at most $2 \times 10^{12}$ trigram occurrences, so at most 6.1% of possible trigrams could ever be seen even once. Possible 4-grams number about $10^{18}$.

### Smoothing on a toy corpus

**Smoothing** is any rule that moves a little probability from sequences that were seen to sequences that were not, so nothing gets exactly zero. **Add-one smoothing**, also called Laplace smoothing, pretends every possible pair occurred once more than it did, which gives $(c(x_{t-1}\, x_t) + 1)/(c(x_{t-1}) + V)$. **Backoff** keeps the observed counts but shaves a discount $\delta$ off each one, then hands the freed probability to unseen continuations in proportion to how common each token is on its own. The table works both after the context "dog" in the twelve-token corpus, where the toy vocabulary has $V = 6$ tokens, the context count is $c(\text{dog}) = 1$, and the discount is $\delta = 0.5$. On its own, "the" occurs 3 times, "cat" 2, "sat" 2, "dog" 1, "ran" 1, and the period 3, out of 12 tokens.

| Next token | Count after "dog" | Raw counting | Add-one | Backoff with $\delta = 0.5$ |
|---|---|---|---|---|
| sat | 1 | 1.000 | 2/7 = 0.286 | (1 − 0.5)/1 = 0.500 |
| ran | 0 | 0 | 1/7 = 0.143 | 0.5 × 1/10 = 0.050 |
| the | 0 | 0 | 0.143 | 0.5 × 3/10 = 0.150 |
| cat | 0 | 0 | 0.143 | 0.5 × 2/10 = 0.100 |
| dog | 0 | 0 | 0.143 | 0.5 × 1/10 = 0.050 |
| . | 0 | 0 | 0.143 | 0.5 × 3/10 = 0.150 |

Both columns sum to one and both rescue "dog ran". Add-one does it bluntly, cutting the one thing the corpus actually showed from 1.0 to 0.286, and with a realistic vocabulary of 32,000 the invented counts would swamp any real ones. Backoff keeps "sat" at 0.5 and spreads the freed half by how common each token is, which is why the strongest counting models, up to Kneser-Ney smoothing developed in Language Models before Transformers, refine discounting combined with backoff or interpolation. Neither rule knows that a dog is like a cat, and that missing knowledge is what the next section supplies.

## Neural language models

A counting model has no notion that two words are alike, since every n-gram is a separate entry in a table. Seeing "the cat ran" therefore tells it nothing about "the dog ran", even though cats and dogs appear in similar sentences. A **neural network** is a function built from many simple arithmetic steps whose behavior is set by adjustable numbers called **parameters**, and a neural language model uses one to compute next-token probabilities instead of looking them up. [Bengio and colleagues](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) introduced the design in 2003, and every modern large language model, or LLM, keeps its three stages.

1. An **embedding** is a learned vector of $d$ numbers stored for each vocabulary token, and the model starts by looking up $e(w)$ for every token in the prefix. The reference 7B model stores $32{,}000 \times 4{,}096 = 131{,}072{,}000$ such numbers in its embedding table. How the lookup works and what the vectors learn is covered in [Tokenization](/tutorials/llm-fundamentals/tokenization-and-embeddings/).
2. The network body combines the prefix embeddings into one **hidden state** $h_t$, a vector of $d$ numbers summarizing everything the prefix says about what comes next. In modern models the body is a stack of transformer layers, blocks that let each position read the earlier positions and then transform what it read, as described in [Transformer Block](/tutorials/llm-fundamentals/the-transformer-block-and-model-anatomy/).
3. An **output head** stores one output vector $u_w$ per vocabulary token and scores each candidate by the dot product $h_t \cdot u_w$, which multiplies matching entries and adds them up. In the reference 7B model the output head is a second, untied table of another 131,072,000 numbers.

![Neural language model stages](/images/llm-fundamentals/15a-neural-language-model-stages.png)

The figure follows the prefix "the cat" through the three stages. Two rows are read out of the embedding table, the network body mixes them into one hidden state, and that state is scored against every output vector, giving the logits that the [logits and softmax](#logits-and-softmax) section turns into probabilities.

**Training** means adjusting the parameters so the model gives higher probability to the tokens that really came next in the corpus. It runs by **gradient descent**, which repeatedly nudges every parameter a small step in the direction that most increases those probabilities. The [cross entropy loss](#cross-entropy-loss) section names the exact quantity being pushed. The gain over counting comes from where the knowledge is stored. A toy model with $d = 2$ shows this, with the hidden state taken to be just the embedding of the previous token. Here training has placed $e(\text{cat}) = (1.0, 0.5)$ and $e(\text{dog}) = (0.9, 0.6)$ close together, because both appear before "sat", "ate", and "slept" in many sentences, while $e(\text{the}) = (-0.4, 1.0)$ sits elsewhere. With $u_{\text{ran}} = (2, 0)$, the score for "ran" after "cat" is $1.0 \times 2 + 0.5 \times 0 = 2.0$, after "dog" it is $1.8$, and after "the" it is $-0.8$.

![Embedding scores](/images/llm-fundamentals/15a-embedding-scores.png)

Every time training raised the score of "ran" after "cat", it moved $u_{\text{ran}}$ toward $e(\text{cat})$, and that same move raised the score after "dog" because the two embeddings are neighbors. Statisticians call this sharing statistical strength, meaning evidence about one context improves predictions for every similar context. That is how the model gives "dog ran" a sensible probability with no count at all. It also means the model needs no table entry per n-gram, so the parameter count stays fixed as the prefix grows from 2 tokens to 4,096.

## Logits and softmax

The scores from the output head can be any real numbers, such as 2.0 or −1.0, but a probability must be between 0 and 1 and the probabilities over all candidates must sum to 1. A **logit** $z_w = h_t \cdot u_w$ is the raw score for candidate $w$ before that conversion. **Softmax** is the function that does the conversion. It raises $e$, about 2.718, to the power of each logit, which makes every value positive while keeping the larger logit larger, and then divides each result by their sum:

$$
P_\theta(w \mid x_{<t}) = \frac{e^{z_w / \tau}}{\sum_{w'} e^{z_{w'} / \tau}}.
$$

Here $z_w$ is the logit for candidate $w$, the sum runs over every vocabulary token $w'$, and $\tau$ is the temperature defined below, equal to 1 during training and scoring. In the context "the cat", four candidates have logits of 2.0 for "sat", 1.0 for "ran", 0.5 for "is", and −1.0 for "the". The exponentials are 7.389, 2.718, 1.649, and 0.368, their sum is 12.124, and dividing gives 0.609, 0.224, 0.136, and 0.030. Only the gaps between logits matter, since adding the same number to every logit multiplies every exponential by the same factor, which cancels in the division. Real code uses that fact to keep softmax numerically stable. It subtracts the largest logit $m$ before exponentiating, and when only the log probability is needed, as in the loss of the [cross entropy loss](#cross-entropy-loss) section, it computes that log directly:

$$
P_\theta(w \mid x_{<t}) = \frac{e^{z_w - m}}{\sum_{w'} e^{z_{w'} - m}}, \qquad \ln P_\theta(w \mid x_{<t}) = (z_w - m) - \ln \sum_{w'} e^{z_{w'} - m}.
$$

Here $m = \max_{w'} z_{w'}$ is the largest logit and $\tau = 1$. Every exponent is now zero or below, so no exponential exceeds 1 and the largest one equals exactly 1, which keeps the sum away from both overflow and zero. Logits of 1,000 and 999, whose exponentials overflow to infinity, become 0 and −1 and give the same weights of 0.731 and 0.269. The second form is called **log-sum-exp**, and it never takes the log of a probability that may have rounded to zero. Mixed-precision implementations usually do this sum in 32-bit precision even when the rest of the model runs in a smaller format.

**Temperature** $\tau$ is a positive number that divides every logit before softmax. Dividing by 0.5 doubles every gap, and the same four candidates become 0.842, 0.114, 0.042, and 0.002, so the top choice dominates. Dividing by 2 halves every gap and flattens them to 0.434, 0.263, 0.205, and 0.097. Sampling means drawing the next token at random according to these probabilities, and temperature is the simplest control over how adventurous sampling is. Decoding rules built on it, such as top-p sampling, belong to [Inference](/tutorials/llm-fundamentals/inference-and-decoding/), and every loss and perplexity in this chapter assumes $\tau = 1$.

<details class="cm-steps not-content" data-anim="15a-softmax-walkthrough">
<summary class="cm-steps-head"><span class="cm-steps-title"><span class="cm-steps-label">Step-by-step example:</span> Softmax</span><span class="cm-collapse-hint" aria-hidden="true"></span><span class="cm-steps-sub">Example from the logits and softmax section: context &quot;the cat&quot;; logits 2.0, 1.0, 0.5, -1.0; temperature divides every logit</span></summary>
<ol class="cm-steps-list">
<li class="cm-step"><p class="cm-step-text"><span class="cm-step-num">Step 1</span>The network ends with one raw score per vocabulary token, called a logit. Logits can be any real number, negative included, and they do not sum to anything in particular.</p><img src="/images/steps/15a-softmax-walkthrough/step-01.png" alt="Step 1 of 7: The network ends with one raw score per vocabulary token, called a logit. Logits can be any real number, negative included, and they do not sum to anything in particular." loading="lazy" decoding="async"></li>
<li class="cm-step"><p class="cm-step-text"><span class="cm-step-num">Step 2</span>At temperature 1 the logits pass through unchanged. Temperature is a number that divides every logit before the next step.</p><img src="/images/steps/15a-softmax-walkthrough/step-02.png" alt="Step 2 of 7: At temperature 1 the logits pass through unchanged. Temperature is a number that divides every logit before the next step." loading="lazy" decoding="async"></li>
<li class="cm-step"><p class="cm-step-text"><span class="cm-step-num">Step 3</span>Exponentiate each one: e to the 2.0 is 7.389, e to the -1.0 is 0.368. Every result is positive and the larger logit stays larger.</p><img src="/images/steps/15a-softmax-walkthrough/step-03.png" alt="Step 3 of 7: Exponentiate each one: e to the 2.0 is 7.389, e to the -1.0 is 0.368. Every result is positive and the larger logit stays larger." loading="lazy" decoding="async"></li>
<li class="cm-step"><p class="cm-step-text"><span class="cm-step-num">Step 4</span>Divide each exponential by their sum 12.124. The probabilities are 0.609, 0.224, 0.136 and 0.030, and they sum to one.</p><img src="/images/steps/15a-softmax-walkthrough/step-04.png" alt="Step 4 of 7: Divide each exponential by their sum 12.124. The probabilities are 0.609, 0.224, 0.136 and 0.030, and they sum to one." loading="lazy" decoding="async"></li>
<li class="cm-step"><p class="cm-step-text"><span class="cm-step-num">Step 5</span>Subtracting the largest logit from all four first gives the same probabilities, because the shared factor cancels in the division. Real code does this so e to a large logit never overflows.</p><img src="/images/steps/15a-softmax-walkthrough/step-05.png" alt="Step 5 of 7: Subtracting the largest logit from all four first gives the same probabilities, because the shared factor cancels in the division. Real code does this so e to a large logit never overflows." loading="lazy" decoding="async"></li>
<li class="cm-step"><p class="cm-step-text"><span class="cm-step-num">Step 6</span>Temperature 0.5 doubles every logit gap before exponentiating. &quot;sat&quot; rises to 0.842 and &quot;the&quot; falls to 0.002, so sampling becomes nearly deterministic.</p><img src="/images/steps/15a-softmax-walkthrough/step-06.png" alt="Step 6 of 7: Temperature 0.5 doubles every logit gap before exponentiating. &quot;sat&quot; rises to 0.842 and &quot;the&quot; falls to 0.002, so sampling becomes nearly deterministic." loading="lazy" decoding="async"></li>
<li class="cm-step"><p class="cm-step-text"><span class="cm-step-num">Step 7</span>Temperature 2 halves every gap. The distribution flattens to 0.434, 0.263, 0.205 and 0.097, so sampling picks the unlikely tokens more often.</p><img src="/images/steps/15a-softmax-walkthrough/step-07.png" alt="Step 7 of 7: Temperature 2 halves every gap. The distribution flattens to 0.434, 0.263, 0.205 and 0.097, so sampling picks the unlikely tokens more often." loading="lazy" decoding="async"></li>
</ol>
<div class="cm-steps-notes">
<p>This example replays the calculation column by column, shows that subtracting the largest logit changes nothing, and then redraws the bars at temperatures 0.5 and 2 over dashed outlines of the temperature 1 answer.</p>
</div>
</details>

## Cross entropy loss

Training needs one number that says how badly the model did on a token, so that gradient descent has something to push down. A **loss function** is that number, chosen so a perfect prediction scores zero and worse predictions score higher. Language models use the **negative log likelihood**, or NLL, which for one token is minus the log of the probability the model gave to the token that actually came next, averaged over the scored tokens:

$$
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \ln P_\theta(x_t \mid x_{<t}).
$$

Here $T$ is the number of tokens scored, $x_t$ is the true token at position $t$, and $x_{<t}$ is the true prefix before it. The minus sign makes the loss positive, since the log of a number below one is negative, and dividing by $T$ makes it an average per token. With natural logs the unit is the **nat**. With base-2 logs the unit is the **bit**, the surprise of one fair coin flip, and one bit equals $\ln 2 \approx 0.693$ nats, so dividing nats by 0.693 converts them to bits.

The loss can be read as surprise. Before each token the model spreads its probability across the vocabulary, and the loss is large when little of that probability went to the token that actually came. A token the model gave probability 1 costs nothing, 0.5 costs 0.69 nats, 0.1 costs 2.30 nats, and 0.01 costs 4.61 nats. The cost grows slowly while the model is roughly right and sharply when it is confidently wrong, and every halving of the probability adds the same 0.69 nats, which is exactly one bit. In "the cat sat" the model gave "the" 0.2, "cat" 0.1 and "sat" 0.3, so "cat" surprised it most, at 2.303 nats, and the three surprises average to 1.705 nats per token. A good language model is one that is rarely surprised by real text.

Papers usually call the same number **cross entropy**. In general, cross entropy compares a target distribution with the predicted one at every scored position and averages over the positions:

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{T}\sum_{t=1}^{T} \sum_{w} y_{t,w} \ln P_\theta(w \mid x_{<t}).
$$

Here the inner sum runs over every vocabulary token $w$, and $y_{t,w}$ is the target weight on token $w$ at position $t$, written $y_w$ when the position is clear. In next-token training the target is **one-hot**, meaning weight 1 on the token that actually came next and 0 on every other token, so the inner sum keeps only the true token's term, $-\ln P_\theta(x_t \mid x_{<t})$, and $\mathcal{L}_{\text{CE}}$ becomes exactly the NLL formula above. The two names part only when the target spreads over several tokens, as in distillation from a teacher model's probabilities in [Post-Training](/tutorials/llm-fundamentals/post-training-from-base-model-to-assistant/).

The word cross refers to the two distributions involved. Reality picks which token comes next, and the model only supplies the probabilities, so cross entropy is the average surprise of a model that predicts with its own probabilities while the data is drawn from the real ones. It is lowest when the two agree, and the [maximum likelihood as distribution matching](#maximum-likelihood-as-distribution-matching) section shows that the gap is exactly the extra cost of the model's mistakes. There is also a compression reading. A coding scheme built from the model's probabilities can store a token it gave probability $p$ in about $-\log_2 p$ bits, so a token at probability 0.5 takes 1 bit and one at 0.01 takes 6.6 bits. Cross entropy in bits is then the average length of the compressed text, and a model that predicts better compresses text into fewer bits, which is the idea behind the bits per byte deep dive in the [perplexity](#perplexity) section.

<details class="cm-collapse">
<summary><span class="cm-collapse-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg></span><span class="cm-collapse-text"><span class="cm-collapse-kicker">Deep dive</span><h3 class="cm-collapse-title" id="why-use-nll">Why use NLL</h3></span><span class="cm-collapse-hint" aria-hidden="true"></span></summary>

NLL is not the only possible loss. Any formula that gets smaller as the model gives the true token more probability could, in principle, be used for training, such as $1 - p$, where $p$ is the probability given to the true token, or a squared error. The running example is the prefix "The capital of France is", with "Paris" at 0.80, "London" at 0.10, "Berlin" at 0.06 and "Rome" at 0.04. NLL charges $-\ln 0.80 = 0.22$ nats for "Paris", and it would charge 0.01 at $p = 0.99$, 0.69 at $p = 0.5$, and 4.61 at $p = 0.01$. Language models use NLL rather than these other losses because of the following five strengths.

1. It is maximum likelihood. By the chain rule of the [chain rule of probability](#the-chain-rule-of-probability) section a text's probability is a product of next-token probabilities, and the log turns that product into a sum, so the summed NLL over a text is exactly minus the log probability of the whole text:

   $$
   -\ln P_\theta(x_1, \ldots, x_T) = -\ln \prod_{t=1}^{T} P_\theta(x_t \mid x_{<t}) = -\sum_{t=1}^{T} \ln P_\theta(x_t \mid x_{<t}).
   $$

   For "the cat sat" that is $-\ln 0.006 = 5.116$ nats, the same total as $1.609 + 2.303 + 1.204$ from the per-token losses in the worked example above. Lowering it raises the probability of the training data, which is the estimation rule the [maximum likelihood as distribution matching](#maximum-likelihood-as-distribution-matching) section develops, and because the sum has one term per token, a batch of tokens is scored by averaging their terms.
2. It keeps numbers in range. A 4,096-token window whose true tokens average probability 0.2 has probability $0.2^{4096} \approx 10^{-2863}$, far below the smallest positive 32-bit floating-point number, about $10^{-45}$, so the product rounds to zero. Its log is $4096 \times \ln 0.2 = -6{,}592$ nats, an ordinary number.
3. Its push does not fade when the model is badly wrong. The **gradient** of the loss is the list of how much it changes when each input moves a little, and through softmax the gradient of NLL with respect to each logit is $\partial \mathcal{L} / \partial z_w = P_\theta(w \mid x_{<t}) - y_w$. For the four candidates in the [logits and softmax](#logits-and-softmax) section with "sat" true that is $(-0.391, 0.224, 0.136, 0.030)$, so gradient descent raises the true token's logit and lowers each other logit in proportion to the probability it wrongly received. With $1 - p$ as the loss, the true logit's gradient is $-p(1 - p)$ instead, so at $p = 0.01$ NLL pushes with $-0.99$ while $1 - p$ pushes with $-0.0099$, a hundred times weaker exactly when the model most needs correcting.
4. It trains faster than squared error. The squared distance between the predicted probabilities and the one-hot target, called the **Brier score**, $\sum_w \left(P_\theta(w \mid x_{<t}) - y_w\right)^2$, is a legitimate loss, but through softmax its gradient also shrinks when the model is confidently wrong. If all the remaining probability sits on one wrong token, the score is $(1-p)^2 + (1-p)^2 = 2(1-p)^2$, which is only 1.96 at $p = 0.01$, compared with an NLL of 4.61. The push on the true token's logit is $-4p(1 - p)^2$, which is $-0.039$ at $p = 0.01$, twenty-five times weaker than NLL's $-0.99$. NLL also grows without bound as $p$ falls, so a model can never safely call an observed token impossible.
5. It rewards honest uncertainty. NLL is a **proper scoring rule**, meaning its average over the data is lowest when the predicted distribution equals the true one, as the [maximum likelihood as distribution matching](#maximum-likelihood-as-distribution-matching) section shows. After "She went to the store to buy some", where "milk", "bread" and "fruit" all occur, the best NLL comes from giving each its real frequency rather than betting everything on one. The Brier score shares this property, but $1 - p$ does not, since its average is lowest when all probability sits on the single most common token.

![Loss against probability](/images/llm-fundamentals/15a-loss-against-probability.png)

The left panel shows the penalty each rule charges, and only NLL climbs steeply on the left, where cutting a probability in half always adds exactly one bit. The right panel shows how hard each rule pushes the true token's logit, and only NLL keeps pushing near full strength as $p$ approaches zero. That shape forces a model trained on NLL to keep some probability on everything that occurs in the data, which the [maximum likelihood as distribution matching](#maximum-likelihood-as-distribution-matching) section turns into a statement about the whole distribution. NLL also has an important limit. It measures how well the model predicts its training text, not whether an answer is true, well reasoned, or what a person wanted, which is why the post-training in [Post-Training](/tutorials/llm-fundamentals/post-training-from-base-model-to-assistant/) adds objectives built on preferences and checked answers.

</details>

## Perplexity

The [cross entropy loss](#cross-entropy-loss) section gave "the cat sat" an average loss of 1.705 nats per token. The number is correct, but few people have a feel for what 1.705 nats means. A count of choices is much easier to picture, and the two are directly linked. A fair six-sided die is the simplest case. The best a model can do when guessing its roll is give each face probability 1/6, so every guess costs $-\ln(1/6) = \ln 6 = 1.79$ nats. The same reasoning works for any number of options. With $k$ equally likely options, each one gets probability $1/k$, so each guess costs $-\ln(1/k) = \ln k$ nats. That gives a way back from a loss to a number of options: the loss is $\ln k$, so raising $e$ to the loss undoes the log and recovers $k = e^{\mathcal{L}}$. For the die, $e^{1.79} \approx 6$, which is the six faces we started with. **Perplexity** applies that conversion to any model. It takes the model's average loss and returns the number of equally likely options that would give the same loss:

$$
\text{PP} = e^{\mathcal{L}} = \left(\prod_{t=1}^{T} \frac{1}{P_\theta(x_t \mid x_{<t})}\right)^{1/T}.
$$

Here $\mathcal{L}$ is the average cross entropy in nats from the [cross entropy loss](#cross-entropy-loss) section, and the second form shows that perplexity is the geometric mean, the $T$-th root of a product of $T$ numbers, of one over each true-token probability. Perplexity is therefore read as an **effective branching factor**, the number of equally likely next tokens the model was effectively choosing between at each step. If losses are in bits, the same number is $2^{\mathcal{L}}$. For "the cat sat", $e^{1.705} = 5.50$, and the second form agrees, since $(1/0.006)^{1/3} = 5.50$. The model was as uncertain as someone picking uniformly among about five and a half tokens. A model that spreads probability evenly over the whole vocabulary gives every token $1/V$, so its loss is $\ln V$ and its perplexity is exactly $V$. For the reference 7B model that is $\ln 32{,}000 = 10.37$ nats and a perplexity of 32,000, and a freshly initialized model whose first logged loss sits far above 10.37 usually has a bug in its initialization or in how the loss is averaged. A loss of 2.3 nats gives $e^{2.3} \approx 10$, the uncertainty of a fair ten-sided die. Perplexity used for evaluation is measured on held out text, meaning text kept out of training, since a model can memorize its training text and look better than it is.

<details class="cm-steps not-content" data-anim="15a-perplexity-walkthrough">
<summary class="cm-steps-head"><span class="cm-steps-title"><span class="cm-steps-label">Step-by-step example:</span> Perplexity</span><span class="cm-collapse-hint" aria-hidden="true"></span><span class="cm-steps-sub">Example from the chain rule of probability section: &quot;the cat sat&quot; with next token probabilities 0.2, 0.1 and 0.3; natural log, so losses are in nats</span></summary>
<ol class="cm-steps-list">
<li class="cm-step"><p class="cm-step-text"><span class="cm-step-num">Step 1</span>At the start of text the model gave &quot;the&quot; probability 0.2. The loss is minus the natural log, -ln 0.2 = 1.609 nats.</p><img src="/images/steps/15a-perplexity-walkthrough/step-02.png" alt="Step 1 of 5: At the start of text the model gave &quot;the&quot; probability 0.2. The loss is minus the natural log, -ln 0.2 = 1.609 nats." loading="lazy" decoding="async"></li>
<li class="cm-step"><p class="cm-step-text"><span class="cm-step-num">Step 2</span>After &quot;the&quot; it gave &quot;cat&quot; only 0.1, so this position costs 2.303 nats, the largest surprise in the sentence.</p><img src="/images/steps/15a-perplexity-walkthrough/step-03.png" alt="Step 2 of 5: After &quot;the&quot; it gave &quot;cat&quot; only 0.1, so this position costs 2.303 nats, the largest surprise in the sentence." loading="lazy" decoding="async"></li>
<li class="cm-step"><p class="cm-step-text"><span class="cm-step-num">Step 3</span>After &quot;the cat&quot; it gave &quot;sat&quot; 0.3, costing 1.204 nats. The total is 5.116 nats and the sentence probability is 0.2 × 0.1 × 0.3 = 0.006.</p><img src="/images/steps/15a-perplexity-walkthrough/step-04.png" alt="Step 3 of 5: After &quot;the cat&quot; it gave &quot;sat&quot; 0.3, costing 1.204 nats. The total is 5.116 nats and the sentence probability is 0.2 × 0.1 × 0.3 = 0.006." loading="lazy" decoding="async"></li>
<li class="cm-step"><p class="cm-step-text"><span class="cm-step-num">Step 4</span>Average the loss over the 3 tokens to get 1.705 nats, then exponentiate. Perplexity 5.50 means the model was as uncertain as a fair choice among about 5.5 tokens at each step.</p><img src="/images/steps/15a-perplexity-walkthrough/step-05.png" alt="Step 4 of 5: Average the loss over the 3 tokens to get 1.705 nats, then exponentiate. Perplexity 5.50 means the model was as uncertain as a fair choice among about 5.5 tokens at each step." loading="lazy" decoding="async"></li>
<li class="cm-step"><p class="cm-step-text"><span class="cm-step-num">Step 5</span>A model that spreads probability evenly over the whole vocabulary pays ln 32,000 = 10.37 nats on every token, so its perplexity is the vocabulary size. Real models start training near this number.</p><img src="/images/steps/15a-perplexity-walkthrough/step-06.png" alt="Step 5 of 5: A model that spreads probability evenly over the whole vocabulary pays ln 32,000 = 10.37 nats on every token, so its perplexity is the vocabulary size. Real models start training near this number." loading="lazy" decoding="async"></li>
</ol>
<div class="cm-steps-notes">
<p>This example builds the running total and running average one token at a time, exponentiates the average, and ends on the uniform-model comparison.</p>
</div>
</details>

Because perplexity is an exponential, equal drops in loss give unequal drops in perplexity. Going from 2.3 to 2.2 nats takes perplexity from 10.0 to 9.0, while going from 1.1 to 1.0 takes it from 3.0 to 2.7.

Perplexity is a standard way to report a language model's loss because it helps in four practical ways.

1. It makes progress readable. Saying a new checkpoint cut the model's choice from about 10 equally likely tokens to about 9 is easier to judge than saying its loss fell from 2.3 to 2.2 nats, so evaluation tables often quote perplexity, even though many training curves plot the loss itself.
2. It gives quick sanity checks. A fresh model should start near a perplexity of $V$, 32,000 for the reference model, and a model scoring near 1.5 on new general text is almost certainly being tested on text it has already seen, because no model is that sure of ordinary writing.
3. It shows how well a model fits a kind of text. Measuring perplexity on held out legal, medical, or code documents tells you how familiar the model is with that domain's language before you spend money adapting it, and a drop after extra training on that domain shows the training worked.
4. It can score data as well as models. Pipelines such as [CCNet](https://arxiv.org/abs/1911.00359) run a small n-gram language model trained on Wikipedia over web pages and sort them into tiers by perplexity, so the pages least like that clean reference text, such as spam and broken pages, can be dropped before pretraining.

Perplexity has two limits. Two models can only be compared on the same held out text with the same tokenizer, which the deep dive below fixes by measuring bits per byte. And a lower perplexity says the model predicts text better, not that it answers questions better, which is why [15i](/tutorials/llm-fundamentals/evaluation-reliability-and-safety/#intrinsic-and-task-evaluation) measures task quality separately.

<details class="cm-collapse" id="bits-per-byte-across-tokenizers">
<summary><span class="cm-collapse-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg></span><span class="cm-collapse-text"><span class="cm-collapse-kicker">Deep dive</span><span class="cm-collapse-title">Bits per byte across tokenizers</span></span><span class="cm-collapse-hint" aria-hidden="true"></span></summary>

Comparing models is one of the most common reasons to measure loss, for example when choosing between two open models or checking whether a new tokenizer helped, and the models being compared often use different tokenizers. That is a problem for perplexity, because perplexity is an average per token and two tokenizers cut the same text into different numbers of tokens. A fine tokenizer that splits words into small pieces makes many easy predictions, such as finishing a word whose first half is already written, and those easy predictions pull the per-token average down without the model being any better. **Bits per byte**, or BPB, divides the total loss by the number of bytes in the raw text instead of the number of tokens, so it charges every model for the same text regardless of how it was cut:

$$
\text{BPB} = \frac{\sum_{t=1}^{T} -\log_2 P_\theta(x_t \mid x_{<t})}{B} = \frac{T \cdot \mathcal{L}}{B \ln 2}.
$$

Here the numerator is the total loss over all $T$ tokens in bits, $B$ is the byte count of the text, $\mathcal{L}$ is the average loss in nats per token, and dividing by $\ln 2$ converts nats to bits. [The Pile](https://arxiv.org/abs/2101.00027) reports its language model results this way for exactly this reason. Two models scoring the 22-byte sentence "the cat sat on the mat" show the difference. Model A uses a coarse tokenizer that produces 6 tokens and pays 12.0 nats in total. Model B uses a fine tokenizer that produces 9 tokens and pays 12.6 nats. Per token, A averages 2.0 nats and a perplexity of $e^{2.0} = 7.39$, while B averages 1.4 nats and a perplexity of $e^{1.4} = 4.06$, so B looks far better. Per byte, A pays $12.0 / (0.693 \times 22) = 0.787$ bits and B pays $12.6 / (0.693 \times 22) = 0.826$ bits, so A is actually better, since it paid less to encode the same sentence.

![Bits per byte](/images/llm-fundamentals/15a-bits-per-byte.png)

At the scale of the reference 7B model, a held-out loss of 1.8 nats per token on English text, with the 32,000-token vocabulary averaging roughly 4 bytes per token, works out to a perplexity of $e^{1.8} \approx 6.0$, or $1.8 / 0.693 / 4 \approx 0.65$ bits per byte. A model with a 128,000-token vocabulary at 5 bytes per token can report a higher per-token loss on the same text even when it compresses the text better, so the comparison has to be done in bits per byte.

</details>

## Maximum likelihood as distribution matching

The [cross entropy loss](#cross-entropy-loss) section trains a model by lowering the average loss on observed tokens, which raises the probability of the data, but it is not obvious that this makes the model's whole distribution resemble real text. That question matters for two reasons. A model is used by sampling from its whole distribution, so a model that ranks the right token first but gets the other probabilities wrong will still write odd text. And knowing how much of the loss could still be removed tells you whether more training or a bigger model is worth paying for. The **data distribution** $P_{\text{data}}$ is the unknown process that produced the corpus, and each corpus sentence is one draw from it. **Maximum likelihood** is the training rule of choosing the parameters that make the observed data most probable, which is the same as minimizing cross entropy on it. Seen at one prefix, the average loss over many draws is the cross entropy between two distributions, and it splits into two parts.

$$
H(P_{\text{data}}, P_\theta) = H(P_{\text{data}}) + D_{\text{KL}}(P_{\text{data}} \,\Vert\, P_\theta).
$$

The last term, the Kullback-Leibler divergence explained just below, is computed as

$$
D_{\text{KL}}(P_{\text{data}} \,\Vert\, P_\theta) = \sum_w P_{\text{data}}(w) \ln \frac{P_{\text{data}}(w)}{P_\theta(w)}.
$$

Here $H(P_{\text{data}}, P_\theta) = -\sum_w P_{\text{data}}(w) \ln P_\theta(w)$ is the expected loss when the truth is $P_{\text{data}}$ and the model predicts with $P_\theta$. **Entropy** $H(P_{\text{data}}) = -\sum_w P_{\text{data}}(w) \ln P_{\text{data}}(w)$ is the loss of a model that knows the truth exactly, the irreducible uncertainty of language itself. **Kullback-Leibler divergence**, or KL divergence, $D_{\text{KL}}(P_{\text{data}} \,\Vert\, P_\theta)$ is the extra loss paid for using the wrong distribution, which is zero only when the two match. The entropy does not depend on $\theta$, so minimizing cross entropy is the same as minimizing KL divergence from the data to the model. The split also answers the second question above. Entropy is a floor that no model can go below, so only the KL divergence part of a loss is room for improvement.

At a prefix where the true next-token distribution over three tokens is $(0.5, 0.3, 0.2)$, the entropy is 1.030 nats, the best loss any model can reach. A model predicting $(0.7, 0.2, 0.1)$ has cross entropy 1.122 nats, so its KL divergence is $1.122 - 1.030 = 0.092$. The definition gives the same number directly:

$$
0.5 \ln\frac{0.5}{0.7} + 0.3 \ln\frac{0.3}{0.2} + 0.2 \ln\frac{0.2}{0.1} = -0.1682 + 0.1216 + 0.1386 = 0.0920.
$$

An overconfident model predicting $(0.98, 0.01, 0.01)$ has cross entropy 2.313 nats and a KL divergence of 1.283, fourteen times worse, even though it puts the most likely token first. It loses because it gave 0.01 to tokens the data produces 30% and 20% of the time. That asymmetry gives maximum likelihood its character. **Mode covering** is the tendency to put probability everywhere the data goes, even at the cost of also putting some where the data never goes, because missing real data costs far more than wasting probability. The opposite direction, $D_{\text{KL}}(P_\theta \,\Vert\, P_{\text{data}})$, is **mode seeking**, which prefers to fit one region of the data well and ignore the rest. Reinforcement learning from human feedback, covered in [Post-Training](/tutorials/llm-fundamentals/post-training-from-base-model-to-assistant/), behaves more like the second, which is one reason assistants sound narrower and more certain than base models.

![Mode covering](/images/llm-fundamentals/15a-mode-covering.png)

The figure fits one bell curve to data with two separate peaks. The maximum likelihood fit sits between the peaks and spreads wide enough to cover both, while the reverse fit locks onto one peak. For text, the covering behavior is why a pretrained model, sampled at temperature 1, sometimes produces strange continuations it has learned are merely possible.

## Teacher forcing and exposure bias

During training, a language model predicts each token from the true prefix in the corpus, never from its own earlier guesses. This is called **teacher forcing**. The loss for "sat" is computed after the true "the cat" even if the model would have written "the dog". Because every prefix is already known, all positions of a sequence can be scored in one parallel pass, so one 4,096-token sequence in the reference 7B model yields 4,095 training predictions at once. Teacher forcing is how large language models are actually trained during pretraining and during [supervised fine-tuning](/tutorials/llm-fundamentals/post-training-from-base-model-to-assistant/#supervised-fine-tuning), the later stage that trains on example answers, and this parallel scoring is the main reason, since generating one token at a time during training would be far too slow over trillions of tokens. Generation works differently. Free-running generation feeds each sampled token back in as part of the prefix, so after the first mistake the model is conditioning on text that came from itself rather than from the corpus. This is how a model writes when people use it, and it is also how the model produces its own training examples during reinforcement learning in post-training, the main training stage that learns from free-running output instead of the true prefix.

![Teacher forcing and generation](/images/llm-fundamentals/15a-teacher-forcing-and-generation.png)

The figure runs both methods on "the cat sat on the mat". Under teacher forcing, the model's top guess after "the" is the wrong word "dog". That costs 2.30 nats, because the true "cat" got only probability 0.10. Position 2 still reads the true "cat", though, so the mistake goes no further, and all five losses come from one pass. Under generation, the same early "dog" is fed back in. Every later step builds on it, and the text becomes "the dog ran to the park".

**Exposure bias** is the mismatch teacher forcing creates, since the model was never trained on prefixes containing its own errors and may not know how to recover from them. If each token had an independent 1% chance of an unrecoverable slip, a 500-token answer would come out clean with probability $0.99^{500} \approx 0.0066$. Scheduled sampling ([Samy Bengio and colleagues, 2015](https://arxiv.org/abs/1506.03099)) addressed it by gradually replacing true prefix tokens with the model's own samples during training. Large modern models mostly do not use scheduled sampling, partly because feeding in the model's own samples breaks the single parallel pass, and the pessimistic calculation overstates the problem for them. Their pretraining text already contains typos, digressions, and corrections, so recovering from an odd prefix is partly learned, and most slips are not unrecoverable. The concern has not vanished, though, since repetition loops under [greedy decoding](/tutorials/llm-fundamentals/inference-and-decoding/#greedy-and-beam-search), studied by [Holtzman and colleagues](https://arxiv.org/abs/1904.09751), and drift over very long reasoning traces are both symptoms of a model feeding on its own output. The main stage of LLM training that does not use teacher forcing is reinforcement learning during post-training, with methods such as Proximal Policy Optimization (PPO) and Group Relative Policy Optimization (GRPO), covered in [Post-Training](/tutorials/llm-fundamentals/post-training-from-base-model-to-assistant/). There the model writes complete answers itself, the answers are scored, and the update is computed on those self-generated prefixes, which directly reduces the exposure-bias gap.

