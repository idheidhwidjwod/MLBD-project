# MLBD Project

A Deep Knowledge Tracing (DKT) system that models student learning across math and German, and extracts a prerequisite graph per subject for curriculum ordering.

## Language

### Learning model

**Topic**:
A distinct learning concept in the curriculum (e.g. "Fraction terms", "Upper and lower case"). The atomic unit a student studies.
_Avoid_: concept, node, category

**Skill**:
A topic at a specific difficulty band. One topic maps to one or more skills (topic × difficulty level).
_Avoid_: exercise, question, item

**Subject**:
One of the two independent curricula: `math` or `german`. Graphs and analyses are always per-subject.
_Avoid_: domain, course

### DKT signal

**Strength**:
The mean absolute change in P(correct) across all target topics when a student answers a source topic, averaged over correct and wrong outcomes: `(Δ_correct + Δ_wrong) / 2`. Measures how much the DKT model reacts to an answer — independent of direction.
_Avoid_: impact, magnitude, weight

**Directionality**:
The per-directed-pair signal `(Δ_correct − Δ_wrong) / strength ∈ [−1, +1]` for a source→target pair. Negative means the source is a prerequisite of the target; positive means the target is a prerequisite of the source.
_Avoid_: direction, asymmetry score

**Prerequisite Score** (`f(A,B)`):
The antisymmetric scalar combining both directed signals: `f(A,B) = (directionality(B→A) − directionality(A→B)) / 2`. `f(A,B) = +1` means A is a strong prerequisite of B; `f(A,B) = −1` means B is a prerequisite of A. By construction `f(A,B) = −f(B,A)` and `f(A,A) = 0`.
_Avoid_: prerequisite weight, edge weight, directional score

### Graph

**Curriculum Graph**:
A directed acyclic graph (DAG) per subject where nodes are topics and a directed edge A→B means "A is a prerequisite of B". Used for curriculum ordering — determining which topics a student should study before others.
_Avoid_: knowledge graph, topic graph, dependency graph

**Prerequisite**:
Topic A is a prerequisite of topic B when mastering A demonstrably improves the model's predicted probability of mastering B, and this relationship is asymmetric (the reverse does not hold equally).
_Avoid_: dependency, requirement, precondition

## Relationships

- A **Subject** has one **Curriculum Graph**
- A **Curriculum Graph** contains **Topics** as nodes and **Prerequisite** relationships as directed edges
- A **Topic** maps to one or more **Skills** (one per difficulty band)
- **Strength** and **Prerequisite Score** are both derived from the DKT model's per-step probability updates

## Example dialogue

> **Dev:** "Should we add an edge between 'Number sets' and 'Fractions' in the graph?"
> **Domain expert:** "Only if the **prerequisite score** is above the per-subject threshold *and* the **strength** clears the reliability floor. A high score with near-zero strength could just be noise — the DKT model barely moved."

## Flagged ambiguities

- "edge weight" was used informally during design — resolved: the canonical term is **prerequisite score** (`f`). Strength is a separate concept used only as a reliability gate, not as an edge weight.
- "connected" was used ambiguously — resolved: the **curriculum graph** tolerates isolated topic nodes. "Connected" is never a construction target; edges are added only when both thresholds are met and acyclicity is preserved.
