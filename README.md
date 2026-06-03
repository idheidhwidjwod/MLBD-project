# Transfer of Learning and Fairness in Dropout Prediction: A Study on Lernnavi

This project was conducted as part of the Machine Learning for Behavioral Data (MLBD) course at EPFL. The data were provided by Lernnavi, a Swiss educational platform online that offers lessons and exercises in German and mathematics for high-school students and recommends tasks based on each student's learning progress.

## Team: R3

- Christophe CHARBONNEL
- Romane VORWALD
- Thybault LORTHIOIS

## Overview & Research Questions

This project investigate 2 separate research questions:

1. RQ1: **To what extent do hierarchical relationships between topics provide evidence of transfer of learning in users’ transaction data?** For example, does mastering linear equations accelerate learning of quadratic equations? To explore this question, we first asked whether prior practice and performance on sibling topics improve the prediction of a user's success when they first enter a new child topic, beyond the student’s general prior ability (`RQ1_XGBoost.ipynb`). We then investigated whether prior mastery of topic A accelerates learning in a related topic B using a Deep Knowledge Tracing (DKT) model, which topic pairs show the strongest transfer effects and whether transfer patterns differ between mathematics and German (`RQ1_DKT_training.ipynb` and `RQ1_DKT_post_analysis.ipynb`).

2. RQ2 *(Ethical question)*: **Does a dropout prediction model produce unequal prediction errors across users’ gender and school track?** This ethical research question investigates whether errors from a model predicting dropout are distributed unevenly across genders and educational groups (`RQ2_ethical_analysis.ipynb`).

## Notebooks Overview

Since 2 complementary approaches were developed for RQ1, the project is organized into several notebooks:

- `RQ1_XGBoost.ipynb` explores whether prior practice and performance on sibling topics improve the prediction of a user's early success when they first enter a new child topic, beyond the student’s general prior experience, engagement, prior accuracy and topic history. One entry event per `(user_id, child_topic_id)` pair is defined using the first 3 eligible `CORRECT`/`WRONG` attempts on the child topic. Two **multiclass XGBoost classifiers** are trained on the same target and are compared: a baseline model using pre-entry student general prior experience, engagement and topics seen features, and a "hierarchical" model that extends this baseline with sibling-history features capturing prior exposure, recency and accuracy on topics sharing the same parent node. The two models are evaluated with macro-F1 and quadratic weighted kappa (QWK) mainly, and feature importance is used to identify the strongest predictors and assess the contribution of sibling-history features. Finally, potential selection bias were investigated to interpret the results.

- `RQ1_DKT_training.ipynb` trains a **Deep Knowledge Tracing (DKT)** model implemented as a per-user LSTM. At each timestep, the model predicts the evaluation of the student's *next* attempt (`WRONG / PARTIAL / CORRECT`) on a given skill. After training, the model is used to extract **pairwise topic-interaction signals** to quantify the strength and directionality of transfer between topic pairs.

- `RQ1_DKT_post_analysis.ipynb` analyzes the results from the trained DKT model obtained after analysis.

**RQ2 (ethical research question)** is investigated by the `RQ2_ethical_analysis.ipynb` notebook. It evaluates whether a dropout prediction model produces unequal prediction errors across users' gender and school track. A **Random Forest classifier** is trained on three week-0 behavioural features (`n_tasks`, `avg_score`, `avg_diff`) to predict whether a student will leave the platform before week 3. Fairness is evaluated across gender and school track (Gymnasium vs. Vocational) using Demographic Parity (chi-square, Cramér's V), Equalized Odds (TPR/FPR gaps with Wilson confidence intervals), the Impossibility Result (Chouldechova, 2017), and **TreeSHAP** explainability to identify which features drive the model's predictions.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/idheidhwidjwod/MLBD-project.git
cd MLBD-project
```

### 2. Create and activate a Conda environment

We recommend using a dedicated Conda environment to avoid dependency conflicts, especially because this project is run mainly through Jupyter notebooks and uses several machine-learning libraries.

```bash
conda create -n mlbd-project-R3 python=3.11 -y
conda activate mlbd-project-R3
```

> The project targets **Python 3.11/3.12** and uses TensorFlow-macOS 2.15 with
> Metal acceleration. On non-Apple hardware replace `tensorflow-macos` and
> `tensorflow-metal` with the standard `tensorflow` package.

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

If Jupyter is not included in your environment, install it as follows:

```bash
pip install notebook ipykernel
```

Then register the Conda environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name mlbd-project-R3 --display-name "Python (R3 MLBD project)"
```

When opening the notebooks, select the kernel named **Python (R3 MLBD project)**.

## Expected Repository Structure

To run the notebooks correctly, the repository should follow the structure below.

**IMPORTANT NOTE:** The data provided by Lernnavi are NOT publicly available. Therefore, the raw data files are not included when cloning the repository. Before running the notebooks, create a `data/` folder at the root of the repository and place the raw files inside it using the filenames shown below.

```text
MLBD-project/
├── RQ1_XGBoost.ipynb           # End-to-end XGBoost pipeline
├── RQ1_DKT_training.ipynb      # End-to-end DKT pipeline
├── RQ1_DKT_post_analysis.ipynb # Pairwise topic-transfer analysis based on trained DKT
├── RQ2_ethical_analysis.ipynb  # Ethical analysis of dropout prediction errors
├── requirements.txt
├── README.md
├── CONTEXT.md
├── data/     # /!\ RAW LERNNAVI DATA: MUST BE ADDED MANUALLY /!\
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
├── src_DKT/                    # Helper scripts for the RQ1 - DKT part
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

## How to Run the Notebooks

After completing the setup and adding the required data files, launch Jupyter from the repository root:

```bash
jupyter notebook
```

Or open the repository root folder and all the notebooks in Visual Studio Code.

Then run the notebooks in the following order:

1. `RQ1_XGBoost.ipynb`

2. `RQ1_DKT_training.ipynb`

3. `RQ1_DKT_post_analysis.ipynb`

4. `RQ2_ethical_analysis.ipynb`

The cells of each notebook must be run in order, and cells should not be re-run independently (in order to avoid silent errors).


## Key Results

### **RQ1 - XGBoost:** Models comparison

Mean metrics across seeds:

|    Model     | Macro-F1  |  QWK    |
|--------------|-----------|---------|
| Baseline     |   0.512   |  0.419  |
| Hierarchical |   0.516   |  0.422  |
| Difference   |   +0.004  | +0.003  |

The hierarchical model slightly outperforms the baseline model across seeds, but the improvement is very small. This suggests that sibling-topic history contains some additional predictive signal, but the effect is weak.

### **RQ1 - XGBoost:** Feature importance analysis on hierarchical model

Top 5 predictors in the hierarchical model:

| Feature | Feature importance score |
|---|---:|
| `num__subject_math` | 0.3129 |
| `cat__child_topic_id_951.0` | 0.0375 |
| `cat__child_topic_id_3110.0` | 0.0311 |
| `cat__child_topic_id_3113.0` | 0.0287 |
| `cat__child_topic_id_3112.0` | 0.0279 |

Top 3 sibling-history features:

| Feature | Feature importance score |
|---|---:|
| `num__prior_sibling_accuracy` | 0.0044 |
| `num__prop_sibling_topics_entered_before` | 0.0040 |
| `num__is_first_topic_in_sibling_group` | 0.0039 |

Sibling-history features have very low importance compared with subject and topic-identity features. This suggests that sibling-topic practice provides only limited additional predictive information beyond general student history and topic-specific effects.

### **RQ1 - XGBoost:** Selection-bias exploration

The notebook also investigates whether students with prior sibling-topic history are comparable to students entering a topic directly. Prepared-entry and direct-entry users differ strongly in prior activity, prior topic exposure, and prior accuracy. Standardized mean differences are especially large for `prior_subject_accuracy` and `prior_global_accuracy`, indicating substantial imbalance between the two groups.

This means sibling-history exposure is not random: it partly reflects user engagement, prior ability, and topic-routing patterns. Therefore, the XGBoost results should be interpreted as predictive evidence, not as causal proof that practicing sibling topics improves early success.


### **RQ1 - DKT:** Model performance (held-out test users, 20% split)

| Subject | AUC    | Accuracy | RMSE   |
|---------|--------|----------|--------|
| Math    | 0.7406 | 61.5 %   | 0.8153 |
| German  | 0.7531 | 62.6 %   | 0.5678 |

Both models use 256 LSTM units, no dropout, trained for 20 epochs. The 1 024-unit
configuration reached only ~0.002 higher AUC during tuning while taking ~4× longer
to train, so 256 units was kept for both subjects.

### **RQ1 - DKT:** Transfer analysis (210 topic pairs per subject)

| Metric | Math | German |
|---|---|---|
| Mean interaction strength | 0.064 | 0.068 |
| Mean \|directionality\| | 0.142 | 0.265 |
| % pairs: A is prerequisite of B | 7.1 % | 38.6 % |
| % complementary (\|d\| < 0.1) | 55.2 % | 22.4 % |

### **RQ2:** Fairness audit of the dropout prediction model

| Attribute | OOB AUC | TPR gap | FPR gap | Equalized Odds |
|-----------|---------|---------|---------|----------------|
| Gender | 0.566 | 0.003 | 0.002 | ✅ Fair |
| School track | 0.566 | 0.030 | 0.028 | ❌ Violated |

n = 18,330 students · overall dropout rate = 47.4% · OOB accuracy = 0.556

**Impossibility Result** (Chouldechova, 2017): applies for school track (base rate diff = 0.028 > 0.02, AUC > 0.5) — Demographic Parity, Equalized Odds, and Predictive Value Parity cannot all be satisfied simultaneously. Does not apply for gender (base rate diff = 0.018).

**SHAP feature importance** (dropout class):

| Feature | RF importance | SHAP mean \|value\| |
|---------|:---:|:---:|
| `avg_diff` | 0.405 | 0.0146 |
| `n_tasks` | 0.335 | 0.0193 |
| `avg_score` | 0.260 | 0.0089 |


Top predictor by SHAP: `n_tasks` (students who attempt more exercises in week 0 are less likely to drop out). `avg_diff` ranks first by RF importance due to high variance but contributes less to individual predictions. Since `avg_diff` is set by Lernnavi's adaptive algorithm, its presence in the model raises a design concern.

**Key findings:**

1. **RQ1 - XGBoost:**

- The hierarchical model produces a small but consistent improvement over the baseline model.
- The effect remains weak: mean macro-F1 improves by about +0.004 and mean QWK by about +0.003.
- Feature importance is dominated by subject and child-topic identity.
- Sibling-history features have very low importance, suggesting limited additional predictive value.
- Selection-bias diagnostics show that students with sibling history are already different from direct-entry students, so the results should not be interpreted causally.

2. **RQ1 - DKT:**

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

3. **RQ2:**

- The model is **fair across gender**: TPR gap = 0.003, FPR gap = 0.002 — errors are symmetric between female and male students.

- The model is **unfair across school track**: Vocational students are falsely flagged as at-risk 2.8 pp more often than Gymnasium students (FPR gap = 0.028). In Switzerland, school track is a socioeconomic proxy — deploying this model as-is would disproportionately target an already more vulnerable group with unnecessary interventions.

- The **Impossibility Result** applies for school track: no model can simultaneously satisfy Demographic Parity, Equalized Odds, and Predictive Value Parity. A deployment decision must explicitly choose which fairness criterion to prioritise.

- **TreeSHAP** identifies `n_tasks` as the main predictor. Since `avg_diff` is controlled by Lernnavi's algorithm rather than the student, the model partially penalises students for a platform decision — a structural concern that should be audited before deployment.
