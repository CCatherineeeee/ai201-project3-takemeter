# TakeMeter

A fine-tuned text classifier that evaluates discourse quality in an online community.

## Community Choice and Reasoning

**Community:** Hacker News comments (news.ycombinator.com).

**Reasoning:**

- **Discourse is varied in quality.** Comments range from reasoned technical arguments, to asserted opinions, to "here's how I did it" experience reports, to one-line meta/social replies. That spread is what makes a discourse-quality classifier meaningful — a community where every comment looks alike would teach the model nothing.
- **The category boundaries are clean.** Because HN comments are short and usually do one thing, distinctions like *opinion vs. argument* or *instruction vs. argument* are observable in the text itself, not a matter of taste. Two annotators reading the same definitions would agree on most examples.
- **Data is trivial to collect.** The Algolia HN Search API (`hn.algolia.com/api/v1`) is public, requires no authentication, returns comment text directly, supports keyword search, and paginates up to 1000 hits per page. Collecting the required 200+ comments is a single short loop, with ample headroom to over-collect and discard low-value items.

**Tradeoff considered:** HN is tech-culture rather than a topical fan community (e.g. anime or sports). I accepted this because the cleaner data and sharper label boundaries make for a stronger classifier project; topic familiarity was not a project requirement.

## Label Taxonomy

Three labels cover substantive HN discourse (junk is filtered out — see Labeling Process). The `reaction` class from the spec's example was dropped — pure emotional reactions are rare on HN. Examples below are verbatim real HN comments (linked by ID).

Labels are defined as a **decision tree** so every comment routes to exactly one, which guarantees mutual exclusivity:

```
1. Does the comment advocate a contested position?
   ├─ No  → explainer        (pure information / first-hand experience, no side taken)
   └─ Yes → 2. Is the position supported with reasoning, evidence, or specifics?
            ├─ Yes → argument
            └─ No  → hot_take
```

The tree keys on *position* and *support*, not on how much explanation a comment contains. (An earlier version defined labels by content type and tested ~38% ambiguous on the `argument`↔`explainer` boundary; see [PLANNING.md](PLANNING.md) §2 for that fix.)

### `argument`
*Advocates a contested position **and** supports it with reasoning, evidence, or concrete specifics.*

- **[#48602957](https://news.ycombinator.com/item?id=48602957):** "The exceptions you're listing are mostly real but much more narrow than you're thinking. Only threats of imminent violence are excepted; it's completely protected by the First Amendment to say things like 'it's getting to a point where we might have to be violent'… Similarly, while defamation as such is illegal, it's not defamatory to ruin someone's reputation by offering mean, unfounded opinions about them. There's simply no general principle that speech which harms people can be restricted." *(takes a position, backs it with specific legal distinctions)*
- **[#48602944](https://news.ycombinator.com/item?id=48602944):** "Price aside, the more important factor is that we don't have the repair infrastructure to make something like this worthwhile yet. For something as critical as a car, we have workshops, spare parts supply chains, and the skilled technicians to do the repairs. Conventional robots require a similar skill set, but you still won't be able to rely on a local repair…" *(claim grounded in concrete infrastructure reasoning)*

### `hot_take`
*Advocates a contested position **without** supporting it with reasoning or evidence.*

- **[#48602965](https://news.ycombinator.com/item?id=48602965):** "That's because half of the apartments aren't being rented to force up prices." *(asserted causal claim, zero support)*
- **[#48602970](https://news.ycombinator.com/item?id=48602970):** "This is why they cane people in Singapore for nuisance crime. Incarceration is expensive and fines don't have the right effect." *(confident policy claim, no backing)*

### `explainer`
*Conveys information or first-hand experience **without advocating a contested position**.*

- **[#48602973](https://news.ycombinator.com/item?id=48602973):** "It's address (de)obfuscation via javascript, meant to prevent harvesting of spam targets via simple scraping. The HTML plain-text reads what I wrote, there's a javascript hook to replace the placeholders with actual e-mail addresses…" *(neutral explanation of a mechanism — no side taken)*
- **[#48602969](https://news.ycombinator.com/item?id=48602969):** "I ended up directly using solvespace's solver instead of the suggested wrapper code since it didn't expose all of the features I needed. I also had to patch the solver to make it sufficiently fast for the kinds of equations I was generating…" *(first-hand 'what I did' report — no contested claim)*

## Data

### Collection Source

Public Hacker News comments via the Algolia HN Search API (`hn.algolia.com/api/v1/search_by_date?tags=comment`) — no authentication, public data only. Script: [collect_comments.py](collect_comments.py).

**Topic scope: AI / programming.** Rather than the full multi-topic HN firehose, collection is restricted to AI/coding threads via topical search queries (`LLM`, `GPT`, `Claude`, `programming`, `Rust`, `compiler`, etc.). Reasons: (1) the discourse-quality labels are topic-independent, so a coherent topic doesn't bias the *form* being measured; (2) consistent vocabulary keeps the dataset readable and annotatable; (3) it stays authentic to HN, where AI/programming is a core subject. The queries still span many sub-topics (models, languages, open source, tooling), so the model can't shortcut on a single keyword.

The pull (12 queries × 100 hits = 1,200 raw) then went through a deliberately *conservative* automated pre-filter — drop empty/deleted, ultra-short (<15 words), inline-URL stripping, and duplicate text — leaving **1,017 substantive candidates** in [hn_comments_to_annotate.csv](hn_comments_to_annotate.csv).

**Targeted collection for class balance.** Early labeling showed `argument` dominating (~72%, over the 70% imbalance threshold) because HN AI/programming threads are debate-heavy. To surface the starved minority classes, two targeted pulls were added: short comments (6–14 words) for `hot_take` (terse one-liners the `<15`-word filter had removed), and Show HN thread comments for `explainer` (where people describe what they built). See [PLANNING.md](PLANNING.md) §4.

Nuanced junk (jokes, pleasantries, bare questions, snark) is marked `junk` by hand during annotation rather than auto-filtered, since rule-based detection of those is unreliable.

### Labeling Process

**Substantive discourse only.** ~40% of HN comments are non-discourse junk (bare questions, jokes, snark, pleasantries, link-drops). These are labeled `junk` and dropped rather than given their own class — the project measures *discourse quality*, which varies among genuine takes, and a junk catch-all would distinguish nothing. Comments were pre-labeled by an LLM against the decision tree, then human-reviewed (see [AI Usage](#ai-usage)). Full annotation rationale and edge-case rules are in [PLANNING.md](PLANNING.md) §2–§3.

### Label Distribution

The final training set ([takemeter_dataset.csv](takemeter_dataset.csv)) is **216 on-topic substantive comments**. (Junk — quips, bare questions, pleasantries — was labeled during annotation, then dropped.)

| label | count | % |
|---|---|---|
| `argument` | 127 | 58.8% |
| `hot_take` | 42 | 19.4% |
| `explainer` | 47 | 21.8% |
| **total** | **216** | 100% |

`argument` is the largest class at 59% — under the assignment's 70% imbalance threshold. Reaching this size and balance took off-topic cleaning plus targeted backfill passes (below).

**Off-topic cleaning.** The first round of collection used broad queries including `programming`, which — because Algolia does prefix matching — also matched `program` (as in "weapons program," "missile program"), pulling in ~40 off-topic comments (Korea defense, healthcare, biology, politics). These were identified and removed. Backfill then used **tight, non-polysemous AI queries** (`fine-tuning`, `transformer model`, `prompt engineering`, `inference`, `embeddings`, `GPU`, `RAG`, `agentic`, `diffusion model`, `hallucination`, `context window`) so new comments are reliably about AI/programming. The labels themselves are topic-independent (they measure discourse *form*), so this cleaning is about honoring the stated AI/programming scope, not about label validity.

Class imbalance is further handled at train/eval time with macro-F1 and class weighting (see Evaluation).

### Difficult-to-Label Examples

All three hard cases sit on the **`argument` ↔ everything-else** seam and are resolved by the [decision tree](#label-taxonomy): (1) does it advocate a contested position? (2) if so, is it supported? Vividness and amount of detail don't decide it — only position and support do.

**1. [#48602981](https://news.ycombinator.com/item?id=48602981) — `hot_take` vs `argument`.**
> "These kind of stories always make me chuckle. The Boston Dynamics videos always show the humanoid robots running through debris, dancing… The reality is however, pushing a parts cart to the other side of the factory, returning and doing it again. 12 hours a day, 7 days a week…"

Vivid and pointed, but it offers no evidence — just a colorfully asserted contrast. **Decision: `hot_take`.** Vividness ≠ support.

**2. [#48602955](https://news.ycombinator.com/item?id=48602955) — `explainer` vs `argument`.**
> "ATProto is good at what it was originally designed for: decentralizing Twitter… When a user decides to jump ship from bsky.social… what does that even mean?… Bluesky slices and dices it to a point where the only reason you would bother considering decentralisation is for ideological reasons."

Reads like an explainer (lots of how-ATProto-works detail), but it clearly *takes and defends a side* — that Bluesky's decentralization is hollow. **Decision: `argument`.** It informs in service of advocating a position.

**3. [#48602956](https://news.ycombinator.com/item?id=48602956) — `explainer` vs `argument`.**
> "Not really, considering there aren't that many app servers. Each PDS sends out its events… The more apps in play, the more connections, but it is not the case that for every new PDS, every other PDS now has to open an additional connection. It scales with the amounts of apps, not PDSs."

Pure mechanism explanation in content, but framed as a rebuttal ("Not really…") that defends a claim against another commenter. **Decision: `argument`** — a defended correction, not neutral exposition.

**What these forced me to sharpen:** the deciding question between `argument` and `explainer` is not *how much detail* a comment has but *whether it defends a contested position*. This `argument`↔`explainer` overlap was the central design risk — and it turned out to be exactly where both the baseline and the fine-tuned model struggle most (see Evaluation).

## Fine-Tuning Approach

**Base model.** `distilbert-base-uncased` — a small (~66M-param) BERT-family encoder, chosen for sequence classification on a small dataset running on a free Colab T4 GPU. A small encoder is appropriate given only ~170 labeled examples.

**Training setup.** 70/15/15 split → 151 train / 32 validation / 33 test. Texts tokenized with the DistilBERT tokenizer; Hugging Face `Trainer`. 10 epochs, batch size 16, learning rate 2e-5, weight decay 0.01, `warmup_ratio=0.1`, evaluation each epoch, `load_best_model_at_end` on macro-F1.

**Hyperparameter decision: class-weighted loss + macro-F1 model selection.** The first training run **collapsed to the majority class** — it predicted `argument` for every test comment (accuracy ≈ the majority-class share, macro-F1 ≈ 0.23). Diagnosis found three causes:

1. `warmup_steps=50` while the whole run was only ~24 steps → the learning rate never finished warming up, so the model barely trained.
2. `metric_for_best_model="accuracy"` → on imbalanced data the always-`argument` model has the highest accuracy, so this actively *selected* the collapsed checkpoint.
3. Plain cross-entropy with no class weighting → the loss is minimized by ignoring the two minority classes.

Fixes: `warmup_ratio=0.1`, `metric_for_best_model="f1_macro"`, and **class-weighted cross-entropy** with inverse-frequency weights (≈ `argument` 0.61 / `hot_take` 1.57 / `explainer` 1.36) via a custom `WeightedTrainer`. This resolved the collapse — the model then predicted all three classes. (A robustness retry at `epochs=20, lr=3e-5` did not improve results, so the reported model uses `epochs=10, lr=2e-5`.)

## Baseline

**Method.** Zero-shot classification with a prompted LLM (Groq), no fine-tuning. Each test comment is sent with the system prompt below and the model's one-word reply is parsed as the label. Evaluated on the held-out test split (33 comments).

**Prompt used** (abridged):

```
You are classifying comments from Hacker News threads about AI and programming.
Assign each comment to exactly one of three categories describing the TYPE of discourse.

Decide using two questions, in order:
1. Does the comment advocate a contested position (take a side)?
   - No  -> explainer
   - Yes -> go to question 2
2. Is that position supported with reasoning, evidence, or concrete specifics?
   - Yes -> argument
   - No  -> hot_take

argument:  Advocates a contested position AND supports it ...   Example: "<real argument>"
hot_take:  Advocates a contested position WITHOUT support ...   Example: "<real hot_take>"
explainer: Conveys info / experience, no side taken ...         Example: "<real explainer>"

Note: length/detail does not decide it — only the two questions above do.
Respond with ONLY the label name. Valid labels: argument / hot_take / explainer
```

The prompt embeds the same decision tree used for annotation (so the baseline is judged on the same rule), with one real example per label. The "length/detail does not decide it" line directly targets the predicted `argument`↔`explainer` confusion.

**Results.** Accuracy **0.939**, macro-F1 **0.94** on the 33-comment test set (31/33 correct).

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| `argument` | 1.00 | 0.89 | 0.94 | 19 |
| `hot_take` | 0.88 | 1.00 | 0.93 | 7 |
| `explainer` | 0.88 | 1.00 | 0.93 | 7 |
| **macro avg** | 0.92 | 0.96 | **0.94** | 33 |

The zero-shot model is strong and balanced across all three classes — notably perfect recall on the two minority classes. This is the bar the fine-tune had to clear.

**Note on the bar.** The zero-shot baseline is extremely strong (0.939). That makes the PLANNING §6 criterion "+0.10 macro-F1 over baseline" effectively unreachable for a small fine-tuned encoder — which, rather than a goalpost to move, is itself the project's central finding (see Evaluation and Reflection). *(Caveat: 33-comment test set, so metrics are noisy.)*

## Evaluation Report

**Headline: the fine-tuned model did not beat the zero-shot baseline — and the gap widened on the larger dataset.** Both are evaluated on the same held-out test set (33 comments).

### Metrics (Both Models)

| | majority-class baseline | zero-shot baseline | fine-tuned DistilBERT |
|---|---|---|---|
| accuracy | 0.576 | **0.939** | 0.667 |
| macro-F1 | — | **0.94** | 0.56 |

Per-class, **zero-shot baseline**:

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| `argument` | 1.00 | 0.89 | 0.94 | 19 |
| `hot_take` | 0.88 | 1.00 | 0.93 | 7 |
| `explainer` | 0.88 | 1.00 | 0.93 | 7 |
| **macro avg** | 0.92 | 0.96 | **0.94** | 33 |

The baseline is strong on *every* class — including `hot_take` and `explainer` at recall 1.00, the exact classes the fine-tune fails on. The large model reads stance and support directly; it doesn't fall back on a length/majority heuristic.

Per-class, **fine-tuned DistilBERT**:

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| `argument` | 0.68 | 0.89 | 0.77 | 19 |
| `hot_take` | 0.60 | 0.43 | 0.50 | 7 |
| `explainer` | 0.67 | 0.29 | 0.40 | 7 |
| **macro avg** | 0.65 | 0.54 | **0.56** | 33 |

Reading them: the fine-tuned model leans toward the majority class — `argument` recall is high (0.89) but minority recall collapses (`hot_take` 0.43, `explainer` 0.29; 5 of 7 explainers are called `argument`). Despite class weighting, the larger `argument` training set pulled predictions back toward `argument`. Scaling the dataset from 169 → 216 did **not** improve the fine-tune (macro-F1 ≈ 0.56, essentially unchanged), while the baseline rose to 0.939.

**Against the PLANNING §6 pass criteria:** ✗ macro-F1 ≥ 0.70 (got 0.56) · ✗ every class recall ≥ 0.60 (`hot_take` 0.43, `explainer` 0.29) · ✗ macro-F1 ≥ baseline +0.10 · ✓ accuracy > majority (0.667 > 0.576). **1 of 4 — fails**, more decisively than the 169-run: the fine-tune trails the zero-shot baseline by **0.27** accuracy. (Caveat: 33-comment test set, so per-class numbers are noisy.)

### Confusion Matrix

Fine-tuned model (rows = true label, columns = predicted label):

| true ↓ \ predicted → | `argument` | `hot_take` | `explainer` |
|---|---|---|---|
| **`argument`** | 17 | 2 | 0 |
| **`hot_take`** | 3 | 3 | 1 |
| **`explainer`** | 5 | 0 | 2 |

The model over-predicts `argument` (25 of 33 predictions). The biggest leak is `explainer` → `argument` (5 of 7 explainers), with `hot_take` → `argument` (3 of 7) close behind. Class weighting prevented a total collapse, but with the larger `argument` class the model still defaults toward `argument` and barely recognizes `explainer`. (`confusion_matrix.png` in the repo is the same data as a plot.)

### Wrong Predictions

**1. `explainer` → `argument`** (confidence 0.39) — *"I could confirm 50-100 demonstrations are enough for fine-tuning pi0/pi05. I did research with aloha and humanoid… It works from 20~40ep… success rate…"*
A first-hand report of experimental findings — information, no contested position (`explainer`). The model called it `argument`, almost certainly because it's dense and technical: it reads detail as `argument`. This is the single most common error type (5 of 11) and the same surface-detail confusion as before, now compounded by the model's bias toward `argument`.

**2. `argument` → `hot_take`** (confidence 0.37) — *"It should not. Abstraction in software engineering brings intelligence. (compression correlates to intelligence)"*
The support is real but compressed into a four-word parenthetical, so the model reads it as a bare assertion. (This exact comment failed the same way in the 169-run — a robust, repeatable failure on terse reasoning.)

**3. `hot_take` → `explainer`** (confidence 0.37) — *"The context window is 16 characters. Talking about tokens per second is meaningless."*
The only error that doesn't involve `argument`. It's a dismissive, unsupported take (`hot_take`), but it's phrased as a flat technical fact, so the model reads it as neutral information (`explainer`). It keys on factual *tone* rather than on whether a position is being asserted.

### Failure Analysis

I pasted all 11 misclassified test comments into an LLM to surface common themes, then verified each by re-reading the examples myself. What held up:

- **The model over-predicts `argument`: 8 of 11 errors are something misclassified *as* `argument`**, and 10 of 11 errors involve `argument` at all. The dominant failure is a majority-class bias, not a balanced confusion.
- **`explainer` → `argument` is the single biggest bucket (5 of 11), and it tracks length/detail.** The mislabeled explainers are dense, technical comments (research findings, "what I built," personal setups) — the model reads detail as `argument`.
- **`argument` → `hot_take` (2 of 11) are comments with compressed support** (a clause or parenthetical); terse reasoning is invisible to the model. The lone `hot_take` → `explainer` error is a dismissive take phrased as a flat technical fact.

What I checked:

- **Confidence carries no usable signal.** Wrong predictions span 0.36–0.43 *and* correct predictions span 0.38–0.46 — the ranges fully overlap, so "defer low-confidence cases to a human" would not work. (In the earlier run I had to *discard* a tentative "errors cluster at low confidence" pattern; the larger run confirms confidence simply doesn't separate right from wrong here.)
- ❌ *Sarcasm as a theme* — none of the misclassified comments are sarcastic; doesn't apply.

The throughline: the model defaults toward the majority class (`argument`) and proxies **surface features (length, technical detail, factual tone)** for the label, instead of the intended rule (*does it take a side, and is it supported?*). Scaling from 169 → 216 examples did not fix this — it deepened the `argument` bias.

### Labeling problem or data problem?

**Mostly a data/model problem, not annotation inconsistency.** The misclassified comments were labeled consistently with the decision tree (a detailed comment that takes a side is `argument`; a short comment with compressed support is still `argument`). Re-checking similar pairs, I didn't find comments labeled differently from each other — so the errors aren't tracing back to inconsistent gold labels. The real causes:

- **A surface + majority-class shortcut minimizes loss.** With ~150 training examples and `argument` the largest class, a model can score well by predicting `argument` for anything dense or technical, *without ever learning "takes a side."* That's exactly what it did (8 of 11 errors are → `argument`).
- **The class prior dominates the signal.** Class weighting reduced but didn't remove this; enlarging the dataset (169 → 216) made the `argument` prior *stronger*, not weaker.

**What I tried that didn't work:** varying the training run (epochs 10 → 12 → 20, lower learning rate) made no meaningful difference, and growing the dataset overall (169 → 216) didn't help either — macro-F1 stayed ≈ 0.56 and the `argument` bias actually worsened, because the extra data was mostly *more `argument`*.

**What would likely help (and what wouldn't):**
- **More minority-class data specifically — `hot_take` and `explainer` — is the most promising lever I didn't fully pursue.** Adding data in general failed because it deepened the majority prior; what the model actually lacks is enough `hot_take` and `explainer` examples to counter that prior. The honest reason I didn't push further: it's hard in *this* community. AI/programming threads are dominated by arguments — substantive, supported positions — so genuinely position-free explainers and unsupported hot takes are comparatively rare to surface, and every candidate has to be read and judged by hand against the decision tree, which makes assembling a *balanced* minority set slow. I expect a targeted, balanced minority collection would help, but it's labor-bound future work I ran out of time for.
- **Hard-case contrast examples** would help a fine-tune specifically: short comments that *are* `argument`s and long, detailed comments that are *not*, to break the length/detail proxy.
- **A different tool entirely** is the most practical fix today: the zero-shot LLM (0.94) already reads stance well, so for this task a strong prompted model beats a small fine-tuned encoder.
- **Not** a tighter definition — the decision tree is already crisp; the model fails to *apply* it, not to understand it.

### Sample Classifications

Five **correctly**-classified test comments run through the fine-tuned model:

| comment (excerpt) | true | predicted | confidence | ✓/✗ |
|---|---|---|---|---|
| "I'm vehemently anti-genAI in creative fields but even I think this metric is stupid and unfair to younger generations…" | `argument` | `argument` | 0.43 | ✓ |
| "The gold standard is code samples. I've got 1000-line convention documents… LLMs sometimes ignore these…" | `argument` | `argument` | 0.42 | ✓ |
| "I set up an inference server so I can hit my own open-weight models from my laptop anywhere…" | `explainer` | `explainer` | 0.41 | ✓ |
| "I love openrouter for this, I just put in $20 and I'm able to chat with almost every model…" | `explainer` | `explainer` | 0.38 | ✓ |
| "If there's one thing I've learned, it's that computer technology is morally evil" | `hot_take` | `hot_take` | 0.40 | ✓ |

**Why the first prediction is reasonable.** The model correctly labels the anti-genAI comment `argument`: it takes a clear contested position — the AI-detection metric is unfair — and supports it (cheating on writing predates genAI; the burden shouldn't fall on younger generations). Position plus support is exactly the `argument` signal.

**But look at the confidences: 0.38–0.43, barely above the 0.33 three-class chance line — even when the model is *right*.** The fine-tuned model is uncertain across the board, so its confidence score can't be used to decide when to trust it or route a case to a human. This is a meaningful limitation for any real deployment.

## Reflection

**The construct I defined is pragmatic; the boundary the model learned is lexical.** My taxonomy encodes a question about what a comment is *doing* in the conversation: does it take a contested position, and does it back that position up? Answering that requires reading stance and intent — recognizing that a paragraph of technical detail can be *neutral exposition* or the *same detail deployed to win an argument*, and that a one-clause parenthetical can be genuine support. That distinction is mostly invisible at the surface. What the model's decision boundary actually settled on is a set of surface correlates of those labels in my small training set.

**What it overfit to: the class prior.** On the 216-example dataset, the model's strongest learned behavior is "when unsure, predict `argument`" — the majority class. It predicted `argument` for 25 of 33 test comments, giving high `argument` recall (0.89) but collapsing the minority classes (`explainer` recall 0.29, `hot_take` 0.43). Class weighting softened the total collapse seen in the first run, but the larger `argument` training set just made the majority pull stronger. Underneath that, it leans on surface correlates (length, presence of exposition) rather than stance.

**What it missed:**
- **Stance itself** — the crux of the taxonomy. It cannot separate "explaining how X works" from "explaining how X works *to argue a point*," so `explainer` is mostly swallowed into `argument` (5 of 7).
- **Calibration.** Even its *correct* predictions sit at 0.38–0.46 confidence, barely above the 0.33 chance line. The model never formed a sharp decision boundary; it hedges toward the prior, so confidence carries almost no signal.

**Why the gap exists.** A small encoder fine-tuned on ~150 examples can minimize loss with a majority-biased, surface-cue heuristic; it feels no pressure to learn the harder pragmatic (stance) construct. Scaling the data 169 → 216 did not help — macro-F1 stayed ≈ 0.56 — and arguably hurt the minority classes by enlarging the `argument` prior. Some irreducible ambiguity is real (a few comments are genuinely 50/50), but the failure is larger than that noise floor.

**Net.** The model learned a weak, majority-leaning heuristic that *correlates* with discourse type, not the discourse-type rule I intended — and it knows it isn't sure (uniformly low confidence). Meanwhile the zero-shot LLM, which already encodes rich language understanding, hits 0.939. The honest conclusion: for a nuanced stance-reading task with a couple hundred examples, fine-tuning a small encoder is the wrong tool — a strong prompted LLM is both more accurate and less work.

## Spec Reflection

**One way the spec helped.** Its hard line on label design — strong vs. weak taxonomies, the mutual-exclusivity test, the 70% imbalance rule — forced the decision-tree taxonomy and surfaced the class imbalance early. That single constraint is why the dataset is clean and the evaluation is interpretable; a vague "good/bad quality" scheme would have produced an unmeasurable mess.

**One way the implementation diverged, and why.** The template assumed a straightforward fine-tune (plain `Trainer`, accuracy-based model selection) that improves on the baseline. The implementation diverged in two ways: (1) it adds a **class-weighted loss and macro-F1 model selection** not in the template, because the default config collapsed to the majority class; and (2) it **reports a fine-tune that lost to the zero-shot baseline** rather than tuning until it "won." The reason for (2) is that a strong baseline plus a small dataset made a genuine win unrealistic, and manufacturing one would contradict the spec's own instruction to honestly assess where the model falls apart.

## AI Usage

AI assistance (Claude) was used throughout; the specific directions and overrides:

1. **Taxonomy design and topic scope.** I directed the AI to derive candidate labels from real comments and stress-test them for mutual exclusivity. I **overrode** its initial "keeping mixed topics is fine" recommendation: after noticing off-topic drift (a query for `programming` had matched "weapons program," etc.), I had it re-scope and re-collect the dataset around AI/programming with tighter, non-polysemous queries.

2. **Annotation assistance (disclosed).** Comments were **pre-labeled by an LLM** against the decision tree, in batches, then **human-reviewed** with attention focused on borderline cases. I corrected mislabels during review — e.g., I reclassified one comment from `hot_take` to `argument` because it cited supporting evidence ("every reply has been complaining…"). The majority of labels were LLM-applied; review concentrated on the `argument`↔`explainer`/`hot_take` borderline calls flagged in the `notes` column.

3. **Training-bug diagnosis.** I directed the AI to diagnose why the first fine-tune collapsed to predicting only `argument`. It identified three causes (warmup longer than the entire run, accuracy-based checkpoint selection rewarding the collapsed model, and no class weighting). I applied the class-weighted fix — but **overrode** the impulse to keep tuning for a baseline-beating number, choosing instead to report the regression honestly.

4. **Failure-pattern analysis.** I directed the AI to surface common themes across the misclassified test comments, then verified each by re-reading. I **discarded/corrected** its tentative "errors cluster at low confidence" pattern after checking the numbers — confidence turned out not to separate correct from incorrect predictions at all — and kept only the patterns that held (see Failure Analysis).
