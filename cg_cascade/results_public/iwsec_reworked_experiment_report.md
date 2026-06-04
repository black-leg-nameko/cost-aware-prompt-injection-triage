# Revised IWSEC Experiment Report

## Label Policy
- Primary task: direct prompt-injection attempt detection.
- Positive label: input contains a prompt-injection attempt, regardless of downstream attack success.
- Negative label: benign input with no instruction-hierarchy attack.
- Excluded: imoxto and other outcome-labeled corpora are not used as semantic ground truth.

## Dataset Audit
| Dataset | N | Label counts | Keyword hit | Duplicate prompts | Role |
|---|---:|---:|---:|---:|---|
| promptshield_train | 18909 | {'0': 9457, '1': 9452} | 35.2% | 0.0% | primary semantic direct prompt-injection benchmark |
| promptshield_validation | 1000 | {'0': 497, '1': 503} | 34.5% | 0.0% | primary semantic direct prompt-injection benchmark |
| promptshield_test | 23516 | {'0': 17030, '1': 6486} | 14.5% | 0.7% | primary semantic direct prompt-injection benchmark |
| deepset_semantic_all | 662 | {'0': 399, '1': 263} | 9.7% | 0.0% | small external semantic benchmark used in prior prompt-injection detector work |
| notinject_hard_negatives | 339 | {'0': 339} | 20.1% | 0.0% | benign hard-negative false-positive stress set |
| lakera_gandalf_attack_only | 1000 | {'1': 1000} | 93.3% | 0.0% | attack-only recall stress set |

## PromptShield Test: Detector Comparison
Full-test classifier rows use all 23,516 PromptShield test prompts. LLM rows use sampled PromptShield test subsets.
| Detector / setting | Acc | Prec | Rec | Spec | F1 | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF LR, validation best-F1 threshold | 0.710 | 0.466 | 0.358 | 0.844 | 0.405 | 14373/2657/4164/2322 |
| TF-IDF LR, validation recall>=0.99 threshold | 0.680 | 0.456 | 0.824 | 0.626 | 0.587 | 10654/6376/1143/5343 |
| OpenAI embedding LR, validation best-F1 threshold | 0.756 | 0.542 | 0.754 | 0.757 | 0.631 | 12891/4139/1594/4892 |
| OpenAI embedding RF, validation best-F1 threshold | 0.657 | 0.417 | 0.613 | 0.674 | 0.497 | 11475/5555/2508/3978 |
| Ayub public RF zero-shot, validation best-F1 threshold | 0.276 | 0.276 | 1.000 | 0.000 | 0.432 | 0/17030/0/6486 |
| LLM judge, balanced 2,000 | 0.808 | 0.945 | 0.654 | 0.962 | 0.773 | 962/38/346/654 |
| LLM judge, natural 3,000 | 0.885 | 0.871 | 0.683 | 0.962 | 0.766 | 2094/83/261/562 |

## PromptShield Same-Subset Comparison
Classifier rows and LLM rows are evaluated on identical sampled PromptShield test prompts.
| Sample | Detector | Acc | Prec | Rec | Spec | F1 | TN/FP/FN/TP |
|---|---|---:|---:|---:|---:|---:|---:|
| Natural 3,000 | TF-IDF LR | 0.724 | 0.496 | 0.375 | 0.856 | 0.427 | 1863/314/514/309 |
| Natural 3,000 | OpenAI embedding LR | 0.767 | 0.555 | 0.757 | 0.770 | 0.640 | 1677/500/200/623 |
| Natural 3,000 | OpenAI embedding RF | 0.668 | 0.424 | 0.586 | 0.699 | 0.492 | 1521/656/341/482 |
| Natural 3,000 | LLM judge | 0.885 | 0.871 | 0.683 | 0.962 | 0.766 | 2094/83/261/562 |
| Balanced 2,000 | TF-IDF LR | 0.610 | 0.707 | 0.376 | 0.844 | 0.491 | 844/156/624/376 |
| Balanced 2,000 | OpenAI embedding LR | 0.755 | 0.757 | 0.751 | 0.759 | 0.754 | 759/241/249/751 |
| Balanced 2,000 | OpenAI embedding RF | 0.631 | 0.645 | 0.586 | 0.677 | 0.614 | 677/323/414/586 |
| Balanced 2,000 | LLM judge | 0.808 | 0.945 | 0.654 | 0.962 | 0.773 | 962/38/346/654 |

## External and Stress Evaluation
| Dataset / detector | Acc | Prec | Rec | Spec | F1 | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|
| deepset, LLM judge | 0.843 | 1.000 | 0.605 | 1.000 | 0.754 | 399/0/104/159 |
| deepset, OpenAI embedding LR | 0.701 | 0.958 | 0.259 | 0.992 | 0.407 | 396/3/195/68 |
| deepset, TF-IDF LR | 0.618 | 1.000 | 0.038 | 1.000 | 0.073 | 399/0/253/10 |
| NotInject hard negatives, LLM judge | 0.982 | 0.000 | 0.000 | 0.982 | 0.000 | 333/6/0/0 |
| NotInject hard negatives, TF-IDF cascade @ validation tau<=0.005 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 339/0/0/0 |
| Lakera Gandalf attack-only, LLM judge | 0.897 | 1.000 | 0.897 | 0.000 | 0.946 | 0/0/103/897 |

## Cascade Results
| Dataset | Policy | Call reduction | LLM call rate | Acc | Rec | Spec | F1 | False bypass |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PromptShield natural 3,000 | fixed_tau_0.01 | 16.9% | 83.1% | 0.888 | 0.683 | 0.965 | 0.769 | 0 (0.0%) |
| PromptShield natural 3,000 | fixed_tau_0.03 | 26.6% | 73.4% | 0.888 | 0.682 | 0.966 | 0.770 | 3 (0.4%) |
| PromptShield natural 3,000 | max_fb_rate_0 | 32.8% | 67.2% | 0.887 | 0.676 | 0.967 | 0.767 | 20 (2.4%) |
| PromptShield natural 3,000 | fixed_tau_0.1 | 43.0% | 57.0% | 0.881 | 0.650 | 0.968 | 0.750 | 64 (7.8%) |
| PromptShield balanced 2,000 | fixed_tau_0.01 | 10.1% | 89.9% | 0.810 | 0.654 | 0.966 | 0.775 | 1 (0.1%) |
| PromptShield balanced 2,000 | fixed_tau_0.03 | 17.2% | 82.8% | 0.810 | 0.654 | 0.967 | 0.775 | 4 (0.4%) |
| PromptShield balanced 2,000 | max_fb_rate_0 | 22.9% | 77.1% | 0.805 | 0.643 | 0.967 | 0.767 | 24 (2.4%) |
| deepset all | fixed_tau_0.01 | 69.5% | 30.5% | 0.751 | 0.373 | 1.000 | 0.543 | 114 (43.3%) |
| deepset all | max_fb_rate_0 | 89.4% | 10.6% | 0.668 | 0.163 | 1.000 | 0.281 | 202 (76.8%) |
| NotInject all | fixed_tau_0.03 | 81.1% | 18.9% | 0.994 | 0.000 | 0.994 | 0.000 | -- |
| NotInject all | max_fb_rate_0 | 90.9% | 9.1% | 0.994 | 0.000 | 0.994 | 0.000 | -- |
| Gandalf all | fixed_tau_0.01 | 7.4% | 92.6% | 0.848 | 0.848 | 0.000 | 0.918 | 74 (7.4%) |
| Gandalf all | max_fb_rate_0 | 23.9% | 76.1% | 0.721 | 0.721 | 0.000 | 0.838 | 239 (23.9%) |

## Shift Detection Fail-Safe
The monitor uses an unlabeled PromptShield natural 1,000-sample calibration window.
The reference window is used only for unlabeled distribution monitoring, not classifier training, threshold tuning, or judge-prompt fitting.
Monitor thresholds are pilot fail-safe thresholds and are not optimized against external labels.
| Stream | Policy | Trigger | PSI | KS p | Keyword drift | Call reduction | Rec./Spec. | F1 | False bypass |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| promptshield_natural_holdout_2000 | tau=0.03 + monitor | -- | 0.01 | 0.734 | -0.006 | 26.6% | 0.675 | 0.760 | 3 (0.5%) |
| deepset_all | tau=0.01 naive | off | -- | -- | -- | 69.5% | 0.373 | 0.543 | 114 (43.3%) |
| deepset_all | tau=0.01 + monitor | PSI,KS | 2.09 | <1e-100 | -0.058 | 0.0% | 0.605 | 0.754 | 0 (0.0%) |
| notinject_all | tau=0.03 naive | off | -- | -- | -- | 81.1% | 0.994 | -- | -- |
| notinject_all | tau=0.03 + monitor | PSI,KS | 3.09 | <1e-100 | 0.007 | 0.0% | 0.982 | -- | -- |
| gandalf_all | tau=0.01 naive | off | -- | -- | -- | 7.4% | 0.848 | 0.918 | 74 (7.4%) |
| gandalf_all | tau=0.01 + monitor | KS,KW | 0.15 | <1e-200 | 0.770 | 0.0% | 0.897 | 0.946 | 0 (0.0%) |

## Mixed Low-Rate Shift Probe
Diagnostic mixtures combine PromptShield holdout traffic with seed-42 Gandalf attack samples.
| Stream | Trigger | PSI | KS p | Keyword drift | Call reduction | False bypass |
|---|---|---:|---:|---:|---:|---:|
| promptshield_holdout_plus_gandalf_1pct | -- | 0.01 | 0.834 | 0.003 | 26.4% | 7 (1.2%) |
| promptshield_holdout_plus_gandalf_5pct | -- | 0.01 | 0.318 | 0.033 | 25.9% | 20 (3.2%) |
| promptshield_holdout_plus_gandalf_10pct | KS | 0.01 | 0.006 | 0.075 | 0.0% | 0 (0.0%) |

## Adaptive Padding Attack Probe
A white-box attacker prepends benign-weighted TF-IDF features to Gandalf attacks; feature terms are redacted.
| Stream | Trigger | PSI | KS p | Keyword drift | Call reduction | False bypass | Adaptive bypass |
|---|---|---:|---:|---:|---:|---:|---:|
| padded_gandalf_only | PSI,KS,KW | 10.51 | 0.000 | 0.770 | 99.9% | 999 (99.9%) -> monitor 0 (0.0%) | 999/1000 |
| promptshield_holdout_plus_adaptive_gandalf_1pct | -- | 0.01 | 0.713 | 0.003 | 27.2% | 23 (4.0%) | 20/20 |
| promptshield_holdout_plus_adaptive_gandalf_5pct | -- | 0.03 | 0.288 | 0.032 | 30.6% | 103 (16.7%) | 100/100 |
| promptshield_holdout_plus_adaptive_gandalf_10pct | -- | 0.07 | 0.022 | 0.073 | 34.0% | 203 (29.3%) | 200/200 |

## LLM False-Negative Audit
| Source | FNs | Top categories |
|---|---:|---|
| promptshield_natural_3000 | 261 | benign-task wrapper or task concatenation: 173; multilingual or non-English cue: 35; explicit override cue missed: 15 |
| deepset_all | 104 | keyword-sparse implicit attempt: 58; multilingual or non-English cue: 25; role-play or output-format ambiguity: 11 |
| gandalf_all | 103 | instruction-probing phrased as benign question: 91; keyword-sparse implicit attempt: 11; benign-task wrapper or task concatenation: 1 |

## Main Findings
- Imoxto should remain excluded from semantic direct-PI detection because its labels are outcome/success-oriented rather than attempt-oriented.
- PromptShield has a strong split shift: TF-IDF LR reaches validation F1 0.940 but only test F1 0.405 at the validation best-F1 threshold.
- OpenAI embedding LR is the strongest non-LLM classifier among the local baselines, but still trails the tuned LLM judge on PromptShield sampled test data.
- The few-shot LLM judge is high precision on PromptShield test: natural-sample precision 0.871, specificity 0.962, F1 0.766.
- Same-subset PromptShield comparisons remove the sampling objection: on natural 3,000 prompts, the LLM reaches F1 0.766 while the strongest local baseline reaches F1 0.640; on balanced 2,000 prompts, the LLM reaches F1 0.773 versus 0.754.
- Conservative tau values are the right primary operating points: tau=0.01/0.03 preserve or slightly improve LLM-only F1 on PromptShield natural while reducing LLM calls by 16.9%/26.6%.
- Aggressive thresholds trade safety for cost: the validation-safe threshold reduces PromptShield natural calls by 32.8% but creates 20 malicious false bypasses (2.4%).
- A lightweight PSI/KS/keyword monitor detects deepset and Gandalf source shift and falls back to LLM-only, removing router-induced false bypass on those streams at the cost of giving up savings.
- A sparse mixed-shift probe confirms the stream-level limitation: 1% and 5% Gandalf mixtures do not trigger fallback, while 10% triggers KS fallback.
- A white-box benign-feature padding attack can push Gandalf attacks into the bypass region; full attack streams trigger fallback, but sparse adaptive mixtures evade the stream-level monitor.
- LLM false negatives are concentrated in benign-task wrappers, keyword-sparse implicit attempts, multilingual prompts, and instruction-probing questions.
- On deepset and Gandalf, even conservative PromptShield-trained router thresholds bypass many malicious prompts. A Best-Paper-quality claim should therefore emphasize robust triage diagnostics and conservative escalation, not unconditional cost reduction.
- NotInject shows the value of a router for benign hard negatives: at tau=0.03, cascade reduces LLM calls by 81.1% while preserving zero false bypass because the set has no positives.
