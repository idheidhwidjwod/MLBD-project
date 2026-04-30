# Knowledge Transfer in E-Learning: A Study on Lernavi

Investigating how mastering one topic influences the learning of related topics
in an e-learning context.

## Team
- Christophe CHARBONNEL
- Romane VORWALD
- Thybault LORTHIOIS

## Overview
This project analyzes data from **Lernavi**, a Swiss e-learning platform for
Math and German. We investigate whether and how a student's learning progress
on one topic transfers to related topics, for example, whether mastering
linear equations accelerates learning of quadratic equations.

We train a **Deep Knowledge Tracing (DKT)** model — a per-user LSTM that, at
each timestep, predicts the evaluation of the student's *next* attempt
(`WRONG / PARTIAL / CORRECT`) on a given skill. After training, we extract
**pairwise topic-interaction signals** from the model to quantify transfer
strength and directionality between topic pairs.

## Research Questions
- Does prior mastery of topic A accelerate learning on related topic B?
- Which topic pairs show the strongest transfer effects?
- Does transfer differ between Math and German?

## Setup

Clone the repository:
```bash
git clone https://github.com/idheidhwidjwod/MLBD-project.git
cd MLBD-project
```

Install dependencies:
```bash
pip install -r requirements.txt
```

> The project targets **Python 3.11/3.12** and uses TensorFlow-macOS 2.15 with
> Metal acceleration. On non-Apple hardware replace `tensorflow-macos` and
> `tensorflow-metal` with the standard `tensorflow` package.

## Repository Structure

```
MLBD-project/
├── DKT_training.ipynb          # End-to-end DKT pipeline (data → model → eval)
├── Post_Analysis.ipynb         # Pairwise topic-transfer analysis
├── requirements.txt
├── data/
│   ├── documents.csv.gz        # Exercise documents (all versions)
│   ├── topic_trees.csv.gz      # Hierarchical topic structure
│   ├── topics_translated.csv   # English topic labels
│   ├── transactions.csv.gz     # Student interaction log
│   ├── users.csv.gz            # User metadata
│   └── study/                  # Study-specific data subset
│       ├── events.csv.gz
│       ├── transactions.csv.gz
│       ├── math_prepost_test.csv
│       ├── presurvey_formatted.csv
│       ├── postsurvey_formatted.csv
│       ├── presurvey_questions.txt
│       └── postsurvey_questions.txt
├── src/
│   ├── config.py               # Global constants (e.g. MASK_VALUE)
│   ├── data.py                 # Data loading and cleaning utilities
│   ├── features.py             # Feature engineering and sequence building
│   ├── models.py               # DKT model definition and training helpers
│   ├── post_analysis.py        # Post-training analysis helpers
│   ├── visualization.py        # Plotting utilities
│   └── debug.py                # Dataset inspection helpers
└── weights/
    ├── math_bestmodel.weights.h5       # Best Math DKT checkpoint
    ├── math_tuning_results.json        # Math hyperparameter search results
    ├── german_bestmodel.weights.h5     # Best German DKT checkpoint
    └── german_tuning_results.json      # German hyperparameter search results
```

## Notebooks

### `DKT_training.ipynb`
End-to-end pipeline: loads and cleans the five raw tables, engineers
interaction sequences, tunes and trains a separate LSTM-based DKT model for
each subject (`math`, `german`), and evaluates performance (AUC, balanced
accuracy). Best weights are saved under `weights/`.

### `Post_Analysis.ipynb`
Loads the trained models and computes **pairwise topic-transfer metrics** over
the full dataset. For each topic pair (A, B) it measures the change in
predicted P(correct on B) triggered by an answer to A, separately for correct
and wrong answers. Two aggregate metrics summarise the result:
- **strength** — overall intensity of the interaction
- **directionality** ∈ [−1, +1] — sign indicates which topic is the
  prerequisite (+1: B is a prerequisite of A; −1: A is a prerequisite of B)

## Data
Data provided by Lernavi. Not publicly available — place the raw files in
`data/` before running the notebooks (see repository structure above for the
expected filenames).

## Results

### Model performance (held-out test users, 20% split)

| Subject | AUC    | Accuracy | RMSE   |
|---------|--------|----------|--------|
| Math    | 0.7406 | 61.5 %   | 0.8153 |
| German  | 0.7531 | 62.6 %   | 0.5678 |

Both models use 256 LSTM units, no dropout, trained for 20 epochs. The 1 024-unit
configuration reached only ~0.002 higher AUC during tuning while taking ~4× longer
to train, so 256 units was kept for both subjects.

### Transfer analysis (210 topic pairs per subject)

| Metric | Math | German |
|---|---|---|
| Mean interaction strength | 0.064 | 0.068 |
| Mean \|directionality\| | 0.142 | 0.265 |
| % pairs: A is prerequisite of B | 7.1 % | 38.6 % |
| % complementary (\|d\| < 0.1) | 55.2 % | 22.4 % |

**Key findings:**

- **Coherence check built in.** A genuine prerequisite relation is asymmetric:
  answering A correctly should predict B's mastery, but not vice versa. The
  `directionality(A→B)` vs `directionality(B→A)` scatter provides a direct
  falsifiability test — pairs near the anti-diagonal are credible prerequisites;
  pairs near the origin are symmetric co-occurring topics.

- **Strength and directionality must be read together.** High directionality at
  near-zero strength means the model barely moves either way; credible prerequisite
  candidates require both high strength and high asymmetry.

- **Math and German show structurally different patterns.** Math has many
  small, well-attempted leaf topics sharing common ancestors — a structure that
  produces clear directional signals. German is flatter and more lexical: the
  majority of topic pairs are complementary rather than prerequisite, which is
  consistent with a domain where many skills are practiced in parallel rather than
  sequentially.
