# DACON — Dependence Aware Evidence Fusion for Auditing Noisy Labels

Code and raw results for the paper:

> G. Arul Dalton and P. Sankar. *Dependence Aware Evidence Fusion for Auditing Noisy Labels.* Submitted to *Machine Learning* (Springer).

DACON estimates, for every training example, the probability that its label is correct. It fuses two views of each example, the mean prediction margin along training (temporal view) and the reliability-weighted agreement of its label with its nearest neighbours in the learned representation (spatial view). Fusion is class-wise in log-odds space, and the spatial evidence is shrunk by the within-component correlation of the two views, estimated by a bivariate Gaussian mixture. The posterior drives soft relabelling of suspect examples.

## Repository layout
| Path | Content |
|---|---|
| `dacon/core.py` | NumPy MLP, robust losses, small-loss selection, DyNeC and DACON (`flags={'dep': True}`) |
| `dacon/benchmarks.py` | data loading, noise models, Co-teaching, JoCoR, simplified DivideMix, SELC, confident learning, ENN |
| `experiments/run_main.py` | all 13 methods × 6 datasets × 4 noise settings × 3 seeds (resumable) |
| `experiments/run_ablation.py` | component ablations |
| `analysis/analyze.py` | average ranks, Wilcoxon tests, label-audit metrics, figures |
| `results/` | raw per-run results used in the paper (`results.jsonl`, `ablation.jsonl`, `summary.json`) |
| `figures/` | all figures (PNG and EPS) |

## Reproduce
```
pip install -r requirements.txt
sh scripts/download_data.sh
python experiments/run_main.py
python experiments/run_ablation.py
python analysis/analyze.py
```
Everything runs on a single CPU core; no GPU or deep learning framework is needed.

## Notes on baselines
All baselines are reimplemented from their papers in one code base with the same networks and training schedule. Co-teaching and JoCoR receive the true noise rate as their forget rate. DivideMix is simplified (no data augmentation; soft cross-entropy for the unlabelled part).

## Licence
MIT
