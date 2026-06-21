# TakeMeter — Planning

A fine-tuned classifier that scores the discourse quality of Hacker News comments. This document is the design record: it addresses the six planning questions, then tracks the build roadmap and open questions.

---

## 1. Community

**Choice:** Hacker News comments (news.ycombinator.com), collected via the public Algolia HN Search API. **Scoped to AI/programming threads** (topical search queries) rather than the full multi-topic firehose — the labels measure discourse *form*, not topic, so a coherent topic doesn't bias them, and consistent vocabulary keeps the data readable/annotatable. Queries span many sub-topics (models, languages, open source, tooling) so the model can't shortcut on one keyword.

**Why this community:** HN's culture explicitly prizes substantive, good-faith discussion, and its guidelines push readers to reward comments that add reasoning or information and to challenge empty assertions. That gives a built-in, community-recognized notion of comment quality to model.

**Why it's a good fit for classification:** the discourse is genuinely varied. A read of 40 real comments turned up reasoned technical arguments, confidently asserted opinions with no backing, neutral how-it-works explanations, first-hand experience reports, plus a long tail of jokes, questions, and snark. That spread is what makes the task non-trivial — a community where every comment looked alike would teach the model nothing. The quality variation is also *legible in the text itself* (presence or absence of supporting reasoning), which is what makes consistent labeling possible.

**Tradeoff accepted:** HN is tech-culture, not a topical fan community (anime, sports). Chosen anyway because the cleaner data and sharper boundaries make a stronger classifier; topic familiarity wasn't required.

---

## 2. Labels

Three mutually exclusive labels covering **substantive** comments (junk is filtered out — see §4). Defined as a decision tree so a comment can route to exactly one label:

```
1. Does the comment advocate a contested position?
   ├─ No  → explainer
   └─ Yes → 2. Is the position supported with reasoning, evidence, or specifics?
            ├─ Yes → argument
            └─ No  → hot_take
```

**`argument`** — *A comment that advocates a contested position and supports it with reasoning, evidence, or concrete specifics.*
- [#48602957](https://news.ycombinator.com/item?id=48602957): "The exceptions you're listing are mostly real but much more narrow than you're thinking. Only threats of imminent violence are excepted… it's not defamatory to ruin someone's reputation by offering mean, unfounded opinions about them…"
- [#48602944](https://news.ycombinator.com/item?id=48602944): "Price aside, the more important factor is that we don't have the repair infrastructure to make something like this worthwhile yet. For something as critical as a car, we have workshops, spare parts supply chains, and the skilled technicians…"

**`hot_take`** — *A comment that advocates a contested position without supporting it with reasoning or evidence.*
- [#48602965](https://news.ycombinator.com/item?id=48602965): "That's because half of the apartments aren't being rented to force up prices."
- [#48602970](https://news.ycombinator.com/item?id=48602970): "This is why they cane people in Singapore for nuisance crime. Incarceration is expensive and fines don't have the right effect."

**`explainer`** — *A comment that conveys information or first-hand experience without advocating a contested position.*
- [#48602973](https://news.ycombinator.com/item?id=48602973): "It's address (de)obfuscation via javascript, meant to prevent harvesting of spam targets via simple scraping…"
- [#48602969](https://news.ycombinator.com/item?id=48602969): "I ended up directly using solvespace's solver instead of the suggested wrapper code since it didn't expose all of the features I needed. I also had to patch the solver to make it sufficiently fast…"

**Design note (v1 → v2):** v1 defined labels by *content present* ("has explanation" vs "has argument"). Testing exclusivity on 40 comments showed ~38% ambiguous, concentrated on `argument` ↔ `explainer`, because HN users explain things *in order to* defend a point. v2 reclassifies by two independent properties (position? / supported?), which is exclusive by construction. Dropped the spec example's `reaction` class — pure emotional reactions are rare on HN.

---

## 3. Hard Edge Cases

The genuinely ambiguous cases all sit on the `argument` boundary:

- **`argument` vs `explainer`** — a comment packed with technical detail that *also* defends a side (e.g. an ATProto explanation that argues Bluesky's decentralization is hollow, [#48602955](https://news.ycombinator.com/item?id=48602955)). The detail makes it *look* like an explainer.
  **Rule when annotating:** ignore how much explanation is present; ask only "is a contested position being defended?" If yes → `argument`. A rebuttal ("Not really…", [#48602956](https://news.ycombinator.com/item?id=48602956)) counts as taking a position.

- **`argument` vs `hot_take`** — a vivid, confident comment that *feels* substantive but offers no actual evidence ([#48602981](https://news.ycombinator.com/item?id=48602981)).
  **Rule when annotating:** vividness and length are not support. Require a real reason, specific, or piece of evidence to call it `argument`; otherwise `hot_take`.

- **Borderline junk vs substantive** — short retorts, asserted one-line advice, rhetorical questions.
  **Rule when annotating:** if it advocates a position or conveys real information, keep it; if it's purely a quip, pleasantry, bare question, or link, filter it.

Operationally: annotation will flag any comment where two labels feel defensible, and those flagged cases get re-decided against the tree and logged. If a *recurring* ambiguity appears that the tree doesn't settle, that is the signal to revise the definitions (or fall back to a 2-label `argument`/`hot_take` set).

---

## 4. Data Collection Plan

**Source:** Algolia HN Search API (`hn.algolia.com/api/v1/search_by_date?tags=comment`), no auth, returns comment text directly, paginates to 1000/page. Pull across several topic queries (and front-page threads) so the data isn't dominated by one subject.

**Pipeline:** over-collect raw comments → drop junk with a pre-filter (pure questions, jokes, pleasantries, link-only, and ultra-short comments under ~15 words) → annotate the survivors against the §2 tree.

**Target counts:** ≥200 labeled comments total, aiming for a usable floor of **~60 per label**. Natural class frequencies are skewed — in the 40-comment sample `argument` dominated, `hot_take` was a minority, and `explainer` was rarest (~13%). So balanced collection requires effort, not just random sampling.

**If a label is underrepresented after 200 examples** (expected for `explainer`, possibly `hot_take`):
1. **Targeted over-collection** — query threads likely to contain the scarce class (e.g. "Show HN", "how does", "I built", technical how-to threads tend to yield `explainer`s) and sample from them specifically.
2. **If still scarce**, accept the imbalance rather than fabricate or stretch labels: keep the natural distribution, document it, and compensate at training/eval time with class weighting and class-stratified metrics (§5).
3. Never relabel a comment to fix balance — that corrupts the signal the whole project depends on.

Label distribution will be reported in the README once annotation is done.

---

## 5. Evaluation Metrics

**Evaluation set:** all metrics are computed on a **held-out test split — 20% of the labeled data, stratified by class, with a floor of ~50 comments**, scored **once** at the end (no peeking during training/tuning). Everything below refers to performance on that split.

Accuracy alone is misleading here because the classes are imbalanced: if `argument` is, say, 55% of the data, a model that *always* predicts `argument` scores 55% accuracy while being useless at the actual job (spotting unsupported takes). So:

- **Macro-averaged F1 (headline metric).** Averages F1 across the three classes equally, so doing well on the dominant class can't hide failure on the minority classes (`hot_take`, `explainer`). This is the number that reflects whether the model learned the *distinctions*, not just the prior.
- **Per-class precision / recall / F1.** Needed because the error costs differ by class. For a community-quality tool, **recall on `hot_take`** matters most — missing unsupported takes defeats the purpose — while **precision on `hot_take`** guards against wrongly flagging good comments. Reporting per class makes these visible.
- **Confusion matrix.** Diagnostic for *which* boundary fails. We expect the model's mistakes to cluster on the known-hard `argument` ↔ `explainer` seam; the matrix confirms whether that prediction holds and where to focus.
- **Comparison vs. two baselines** — a majority-class baseline (accuracy = largest class share) and a zero-shot prompted baseline. The fine-tuned model has to beat both, or it isn't earning its keep.

- **Inter-annotator agreement (human ceiling).** Double-label **~50 comments** and report **Cohen's κ**. This sets a realistic ceiling — the model is not expected to exceed human agreement on an inherently fuzzy boundary — and a low κ is itself a finding (the labels, not the model, are the problem).

---

## 6. Definition of Success

**Pass criteria (objective, checked on the §5 held-out test set):**

1. **Macro-F1 ≥ 0.70**
2. **Recall ≥ 0.60 for every class** (no class left behind)
3. **Macro-F1 at least +0.10 above the zero-shot prompted baseline**
4. **Accuracy above the majority-class baseline**
5. Cohen's κ reported as the human ceiling for context

All four numeric thresholds must hold to call it a pass. At that level the model carries real signal on every class rather than leaning on the majority prior.

**"Good enough" for deployment depends on the use:**
- **As an assistive / human-in-the-loop signal** (surfacing or sorting comments by likely quality, suggesting "this looks like an unsupported claim"), the bar above is acceptable — a wrong call is cheap because a human stays in the loop.
- **As autonomous moderation** (auto-hiding or auto-flagging with no human), the bar is much higher and probably out of reach for a first pass, especially given the fuzzy `argument`/`explainer` boundary; I would *not* claim it's ready for that.

So the honest success target is a useful **soft signal**, not an autonomous judge — and the reflection will report which of these the trained model actually reached.

---

## AI Tool Plan

This is a label-design and data project, not an implementation project — there's little code to generate, so AI tools help at three specific points instead.

### 1. Label stress-testing (before annotation)

Give the AI (Claude) the §2 decision tree, definitions, and §3 edge cases, and ask it to generate **5–10 comments deliberately engineered to sit on a boundary** — half on `argument` ↔ `explainer`, half on `argument` ↔ `hot_take`. Then classify each myself using the tree.
- **Pass:** every generated comment resolves to exactly one label via the tree.
- **Fail:** if any comment is genuinely unclassifiable, the definitions are too loose — **tighten them now**, before annotating 200 examples, and re-run the test.
- This happens *before* §6 collection. Any definition change is logged as a taxonomy version bump (v2 → v3) with the reason.

### 2. Annotation assistance

**Decision: hybrid, human-first.** Annotate the **first ~75 comments entirely by hand** to internalize the boundaries, *then* allow LLM pre-labeling for the remainder — never the reverse (pre-labeling first risks rubber-stamping the model's guesses, which corrupts the exact signal this project is about).
- **Tool:** Claude, prompted with the §2 tree, one comment at a time (or small batches), returning a label.
- **Tracking (for disclosure):** the dataset carries a `provenance` column — `human` vs `llm_prelabeled_then_reviewed`. Every LLM pre-label is reviewed and corrected by hand before it counts; the column records origin, not final authority.
- **Bias check:** measure LLM-vs-final-human agreement on the pre-labeled batch and report it. A high override rate is a signal the pre-labels were anchoring me and is worth disclosing.
- Disclosed in full in the README **AI Usage** section.

### 3. Failure analysis (after evaluation)

Feed the AI the list of **wrong predictions** (comment text, true label, predicted label, confidence) and ask it to propose patterns *before* I write the evaluation.
- **What to look for:** which boundary the errors cluster on (hypothesis: `argument` ↔ `explainer`, per §3); correlation with comment length, topic/domain, or hedging language; whether high-confidence errors share a trait (a sign of a systematic blind spot, not noise).
- **How to verify (do not trust the AI's patterns blindly):** for each proposed pattern, go back to the actual wrong examples, **count how many genuinely fit**, and cross-check against the confusion matrix. Keep only patterns that hold quantitatively; discard plausible-sounding ones that don't. The write-up reports verified patterns with counts, not the AI's raw claims.

## Roadmap

1. ~~Choose community~~ ✓ Hacker News
2. ~~Read 30–40 real comments, find patterns~~ ✓
3. ~~Draft labels + clear/uncertain examples~~ ✓
4. ~~Test mutual exclusivity; fix overlap~~ ✓ (v2 decision tree)
5. ~~Write this planning doc~~ ✓
6. Collect + junk-filter + annotate ≥200 comments; report label distribution.
7. Baseline: zero-shot prompted classifier; record prompt + method.
8. Fine-tune (base model + training setup + ≥1 hyperparameter decision).
9. Evaluate both models per §5; confusion matrix; 3 wrong-prediction analyses; sample-classifications table.
10. Reflection, spec reflection, AI-usage sections.

## Open questions

- Does `explainer` survive annotation, or does it overlap `argument` enough in practice to collapse to a 2-label set?
- Final class balance, and how aggressively to over-collect the minority classes vs. lean on class weighting.
- Base model choice for fine-tuning (decide at step 8).
