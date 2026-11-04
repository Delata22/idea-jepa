# Idea-JEPA — ideas above tokens

A research prototype of a **two-level language model**: the model first predicts the *idea* of the next sentence in a
latent space, then a small, deliberately short-sighted decoder turns that idea into words. Built from scratch in PyTorch
(encoder, predictor, memory, decoder, training loop, evaluation, baselines) over about one week of intensive work,
plus a cleanup pass to close the first benchmark.

> Status: **bAbI tasks 1-3 closed** (see [Results](#results-babi-tasks-1-3)). This is a personal research project, not a
> state-of-the-art claim: the architecture combines known ingredients, and bAbI is a small, synthetic benchmark.

## The question

Language models predict the next *token*. People plan what to say at a coarser level (Levelt's *conceptualisation*,
then *formulation*). Can a model learn, **without any labels**, a representation of sentences ("ideas") good enough to
reason over them at the sentence level, and then express them in words?

## Architecture

```
 sentence 1    sentence 2    sentence 3            sentence 4 (true)
     │             │             │                       │
   [ E ]         [ E ]         [ E ]                 [ E_ema ]  (slow copy of E, no gradient)
     │             │             │                       │
    Z_1           Z_2           Z_3                     Z_4  ◄──── JEPA loss ────┐
     │             │             │                                               │
     └──── write ──┴──► delta memory S ───┐                                      │
                   │             │        │ read at every layer                  │
                   └── 2-idea window ─────┴──► [ P : causal idea predictor ] ──► Ẑ_4
                                                                                 │
                                                       [ D : local decoder ] ◄───┘
                                                                 │
                                                       "so the football is in the garden ."
```

| component | what it does |
|---|---|
| **Idea Z** | `r` key/value slots `(k_i, v_i)`, with `‖k_i‖ = 1`: a tensor-product representation `Z = Σ v_i k_iᵀ` (role → content). Roles are *not* labelled; they are learned. |
| **Encoder E** | Bidirectional transformer over a sentence's tokens, then `r` learned queries cross-attend to it (Perceiver-style), giving the slots. |
| **Predictor P** | Causal transformer over *slot tokens* only (it never sees a word). It attends directly to the **last 2 ideas**; everything older goes through a **delta-rule memory** `S ← S(I − β k kᵀ) + β v kᵀ`, written once per idea and read at every layer. |
| **Decoder D** | Sees only the current sentence's tokens (+ 3 tokens of context) and the predicted idea. It reads the slots by attention and, after each word, **erases** from the slots what that word expressed: `v_i ← v_i − g·a_i·(eᵀv_i)·e`. Everything it knows about the story has to go through the idea. |

**Training (no labels from the data generator).** E, P and D are trained jointly from scratch with:

- `L_tok`: cross-entropy of the decoder, both from the encoded idea (reconstruction path) and from the predicted idea
  (generation path, the real language model; its share ramps from 0 to 50 %);
- `L_jepa`: distance between the predicted idea and the target idea from an EMA copy of E (stop-gradient), as in I-JEPA;
- anti-collapse terms: VICReg on the ideas of a batch, and InfoNCE with positives made by dropout (two views of the same
  sentence) and negatives from other stories.

**What made it work on bAbI** (measured in ablations during development): a short window plus a delta memory instead of full
attention over all past ideas (full attention memorises the training stories: 100 % train, flat validation), and depth
in P (8 layers) for the two-hop composition *object → carrier → place*.

## Results (bAbI tasks 1-3)

**Data.** bAbI v1.2 (Weston et al., 2015), English, 10k training questions per task; validation = last 10 % of the
official train set; test = official 1000 questions. One idea per sentence; the question is an idea; the answer is
rewritten as a templated sentence (`where is the football ?` → `so the football is in the garden .`).

**Protocol (identical for every model).** Free greedy generation of the answer sentence; exact match of the answer word.
Snapshot chosen on validation (every 1000 steps), then test. Same budget for all models: 6000 steps of 8192 tokens.
Success criterion fixed in advance: **test error < 5 % on each of 3 seeds**.

**Models.**
- *Idea-JEPA, minimal*: d = 64, 4 slots, E 1 layer, D 2 layers, P 8 layers (window 2 + delta memory): **0.92 M** parameters.
- *B0*: standard causal transformer over tokens, full context, 8 layers, d = 88: 0.93 M.
- *B3*: token-level transformer with a 16-token sliding window + DeltaNet memory written at **every token**
  (i.e. the same memory, without the idea level), 8 layers, d = 76: 0.95 M.
- References: learning rate chosen among {1e-3, 3e-4}, then dropout among {0, 0.1, 0.3}, both on qa2 validation, seed 0
  (chosen: 1e-3, dropout 0.3). On qa3 the references use gradient accumulation (same 8192 tokens per step) to fit in
  memory; B3 seed 2 on qa3 was not run (time). Idea-JEPA uses lr 1e-3, not tuned per task.

**Test error** (1000 official test questions; `-` = not run):

| model | params | task | seed 0 | seed 1 | seed 2 | mean | seeds < 5 % |
|---|---|---|---|---|---|---|---|
| Idea-JEPA (minimal, ours) | 0.92 M | qa1 | 0.6 % | 0.0 % | 0.0 % | **0.2 %** | **3/3** |
| Idea-JEPA (minimal, ours) | 0.92 M | qa2 | 0.6 % | 2.8 % | 18.8 % | **7.4 %** | **2/3** |
| Idea-JEPA (minimal, ours) | 0.92 M | qa3 | 71.1 % | 75.7 % | 70.5 % | 72.4 % | 0/3 |
| B0 transformer | 0.93 M | qa1 | 46.2 % | 48.8 % | 48.3 % | 47.8 % | 0/3 |
| B0 transformer | 0.93 M | qa2 | 81.8 % | 81.0 % | 82.5 % | 81.8 % | 0/3 |
| B0 transformer | 0.93 M | qa3 | 84.7 % | 78.5 % | 78.5 % | 80.6 % | 0/3 |
| B3 window 16 + delta memory | 0.95 M | qa1 | 28.2 % | 28.0 % | 36.5 % | 30.9 % | 0/3 |
| B3 window 16 + delta memory | 0.95 M | qa2 | 76.3 % | 81.8 % | 79.7 % | 79.3 % | 0/3 |
| B3 window 16 + delta memory | 0.95 M | qa3 | 80.3 % | 84.4 % | - | 82.3 % | 0/2 |
| B0, no dropout | 0.93 M | qa1 | 49.4 % | 51.3 % | 52.8 % | 51.2 % | 0/3 |
| B0, no dropout | 0.93 M | qa2 | 83.6 % | 81.3 % | 82.2 % | 82.4 % | 0/3 |
| B3, no dropout | 0.95 M | qa1 | 19.7 % | 27.0 % | 29.6 % | 25.4 % | 0/3 |
| B3, no dropout | 0.95 M | qa2 | 81.6 % | 80.2 % | 79.5 % | 80.4 % | 0/3 |

qa3, Idea-JEPA, test error by number of times the asked object moved (no test question has a single move):

| seed | 2 moves | 3+ moves |
|---|---|---|
| 0 | 70.3 % (n=266) | 71.4 % (n=734) |
| 1 | 79.7 % (n=266) | 74.3 % (n=734) |
| 2 | 66.2 % (n=266) | 72.1 % (n=734) |

Full table, regenerated by `scripts/babi_results_table.py`: [`reports/babi_results.md`](reports/babi_results.md).

## What this does and does not show

**Where it fails.**
- **qa3 (three supporting facts) is not solved**: 72 % error on all 3 seeds, far from the criterion. The error does not
  depend on how many times the object moved (2 or 3+), so the model does not even handle the shortest qa3 stories.
  The references fail too (B0 81 % over 3 seeds, B3 82 % over 2).
- **qa2 passes on 2 seeds out of 3 only.** Seed 2 stays at 18.8 %: the outcome depends on the seed (two basins, one that
  learns the rule and one that memorises, also seen in earlier development runs), so the criterion "< 5 % on each of 3
  seeds" is not met.

**What is not comparable.**
- The references are size-matched (about 0.93 M parameters) and budget-matched, but only lightly tuned: two learning
  rates and three dropout values, on qa2 seed 0. Small transformers trained from scratch are known to do poorly on bAbI,
  so their failure here is expected. It does not show that Idea-JEPA beats a well-tuned baseline.
- Published memory networks (for example MemN2N on the 10k set) already solve qa1-3 under 5 %. Nothing here is a
  state-of-the-art claim.

**What it does show, with these reservations.**
- At equal size and budget, the idea-level recipe solves qa1 (3/3 seeds) and mostly qa2, where both token-level
  references stay at 30-80 % error. B3 has the same delta memory as Idea-JEPA but writes it at every token instead of
  once per sentence. The gap between the two points to the idea level (write once per sentence, read locally) as the
  useful bias, not the memory alone.
- Dropout does not rescue the references (B0 qa1: 51 → 48 %; B3 qa1: 25 → 31 %).

**Not tested.** Scale: whether the advantage over full attention survives 10x more data. EntNet-style biases have
historically lost to attention as data grows, so this is the decisive next test.

## Related work (no single component is new)

- Slots as role/content bindings: tensor product representations (Smolensky, 1990), Perceiver (Jaegle et al., 2021).
- Predicting at the sentence level: Large Concept Models (Meta, 2024); Hwang et al. 2025, *Latent Reasoning via Sentence
  Embedding Prediction* (arXiv 2505.22202).
- Joint-embedding prediction with an EMA target: I-JEPA (Assran et al., 2023); VICReg (Bardes et al., 2022).
- Delta-rule memory: fast weight programmers (Schlag, Irie & Schmidhuber, 2021), DeltaNet (Yang et al., 2024); the
  "write once per sentence into a state, read locally" bias is the one of EntNet (Henaff et al., 2017), which solved
  bAbI in 2017.
- Published concept-level models freeze a pretrained sentence encoder; here E, P and D are learned jointly from scratch.

The combination (key/value idea slots + delta memory at the idea level + JEPA/EMA target + erasing local decoder) is,
as far as I could check, not published; the individual mechanisms are.

## Reproduce

```bash
uv sync                                  # PyTorch 2.x + CUDA (tested on an RTX 5070 Ti, 12 GB)
bash scripts/download_babi.sh            # bAbI v1.2 (Keras mirror, checksum verified)
uv run python -m ideajepa.data.reasoning # synthetic data used by part of the tests (about 2 min)
uv run pytest -q                         # 295 unit tests (4 skipped without a free GPU or opt-in flags)
uv run python scripts/prepare_babi.py    # data/public/babi -> data/babi/qa{1,2,3}_{1k,10k}/{train,val,test}.pt
bash scripts/reproduce_babi.sh           # 9 Idea-JEPA runs + 18 tuned and 12 no-dropout reference runs, then the table
uv run python scripts/babi_results_table.py
```

One Idea-JEPA run takes about 15 minutes on one GPU; a reference run 5 to 60 minutes (B3 on qa3 is the longest). Configs: [`configs/babi_min.yaml`](configs/babi_min.yaml),
[`configs/babi_b0_small.yaml`](configs/babi_b0_small.yaml), [`configs/babi_b3_small.yaml`](configs/babi_b3_small.yaml);
any key can be overridden with `--set section.key=value`.

## Repository layout

```
src/ideajepa/    model (encoder, predictor + delta memory, erasing decoder, baselines), losses, data, training
configs/         one YAML per experiment, with inheritance (base:); bAbI: babi_min, babi_b0_small, babi_b3_small
scripts/         data download and preparation, bAbI evaluation, diagnostics (memory probes, error breakdowns), tables
tests/           unit tests (erasure invariants, causality masks, memory rule, checkpoint resume, ...)
reports/         final results table
runs/            checkpoints, logs and evaluation outputs (created by the scripts, not versioned)
```

## Project history

The project went through a synthetic toy language (stage 1: does the erasing decoder work at all?), a synthetic
"reasoning" generator (v2) where the idea level was compared with strong token-level references, a Path-Star planning
side study, and then public data, starting with bAbI. Many early positive results did not survive tuned baselines or
more seeds; only the bAbI results above are claimed. The code for the earlier stages (synthetic generators, Path-Star,
plan and flow-head variants) is kept in `src/` and `configs/` for completeness.

## About this code

Implemented with the help of AI coding assistants; the research direction, experimental design and analysis are mine.

## License

MIT (see [LICENSE](LICENSE)). bAbI is distributed by its authors under CC BY 3.0 and is not included in this repository.
