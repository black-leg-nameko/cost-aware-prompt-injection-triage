# Sparse mixture audit-on end-to-end

Mixture: PS holdout + 1% padded Gandalf (seed 42).
Audit: 5.0% of bypass-eligible prompts (seed 99).

| Metric | Value |
|---|---:|
| Adaptive attacks | 20 |
| Adaptive audited | 2 |
| Adaptive caught when audited | 2 |
| Adaptive E2E FN | 18/20 (90.0%) |
| Malicious bypass audited | 2 |
| Malicious caught when audited | 2 |
| All malicious E2E FN rate | 3.3% |
| LLM call rate | 77.1% |
| Call reduction vs LLM-only | 22.9% |
| Naive call reduction (no audit) | 23.9% |
