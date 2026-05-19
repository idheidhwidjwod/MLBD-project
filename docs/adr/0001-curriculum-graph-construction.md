# Curriculum graph built by greedy acyclic edge addition with a strength floor

We build the per-subject curriculum DAG by (1) discarding edges below the 50th-percentile strength floor, (2) ranking survivors by prerequisite score descending, and (3) adding each edge greedily unless it creates a cycle. We process all edges above both thresholds — there is no "stop when connected" condition.

## Considered options

**Composite ranking (f × strength):** Rejected because strength and prerequisite score measure independent things — reliability of the DKT signal vs. directionality of the prerequisite relationship. Multiplying them conflates the two: a strongly directional but low-observation pair and a high-observation but symmetric pair would score identically, which is wrong for curriculum ordering.

**Threshold-only (no cycle check):** Rejected because threshold filtering alone does not prevent 3-cycles (2-cycles are impossible by the antisymmetry of f, but A→B, B→C, C→A can each have positive f). A cyclic prerequisite graph is meaningless for curriculum ordering.

**Stop when connected:** Rejected. Connectivity is not a correctness criterion — some topics may have no strong prerequisite signal and should remain isolated rather than receiving a spurious edge to force connectivity.

## Consequences

- Thresholds are per-subject percentiles (not shared absolute values), because math (54% complementary pairs) and german (25% complementary pairs) have structurally different signal distributions. A single cutoff would under-include for german or over-include for math.
- Isolated topic nodes are valid output. They mean the DKT found no reliable directional signal for that topic, not a bug.
- The 50th-percentile values (strength floor, then f ranking) are starting points chosen to yield ~21 edges over 15 nodes. They should be revisited if the topic scope is expanded beyond the top-15.
