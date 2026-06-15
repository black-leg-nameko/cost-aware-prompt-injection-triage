# NP/Clopper-Pearson Source-Matched Triage Calibration

Calibration source: PromptShield official test rows excluding all LLM-judged natural/balanced evaluation records.
Calibration n=10000, labels={'0': 7440, '1': 2560}.
The threshold is the positive-score order statistic whose one-sided upper tolerance bound is below epsilon.

| Policy | eps | alpha | tau | cal FB | cal CP upper | cal call red. | natural call red. | natural FB | natural F1 | balanced call red. | balanced FB | balanced F1 | non-cal test FB | unused holdout FB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| np_cp_eps_0_005_alpha_0_05 | 0.005 | 0.05 | 0.02081 | 6/2560 | 0.5% | 21.8% | 23.2% | 1/823 (0.1%) | 0.769 | 15.2% | 2/1000 (0.2%) | 0.775 | 6/3926 (0.2%) | 3/2229 (0.1%) |
| np_cp_eps_0_01_alpha_0_05 | 0.010 | 0.05 | 0.02918 | 17/2560 | 1.0% | 25.3% | 26.4% | 3/823 (0.4%) | 0.770 | 17.0% | 3/1000 (0.3%) | 0.775 | 15/3926 (0.4%) | 9/2229 (0.4%) |

## Calibration positive budget

| Allowed calibration FB | Minimum positives for eps=0.005, alpha=0.05 |
|---:|---:|
| 0 | 598 |
| 1 | 947 |
| 2 | 1258 |
| 5 | 2100 |
| 6 | 2366 |
| 10 | 3389 |
| 20 | 5808 |

## Hash split sensitivity

| Selection half | Tau | Select FB | Confirm FB | Confirm CP upper | Confirm pass |
|---|---:|---:|---:|---:|---|
| 0 | 0.01867 | 2/1273 | 0/1287 | 0.2% | yes |
| 1 | 0.02032 | 2/1287 | 3/1273 | 0.6% | no |
| min_of_hash_halves | 0.01867 | -- | 2/2560 | 0.2% | yes |
