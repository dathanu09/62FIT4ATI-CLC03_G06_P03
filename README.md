# DataSEC 2026 (P03): Environmental Sound Classification under Acoustic Perturbations

Group G06 · Course: Advanced Topics in Information Technology · Hanoi University

- **GitHub repository:** [TO FILL: repository URL]
- **Model checkpoints (`ckpt_*.pt`, not included in the zip):** [TO FILL: Google Drive link, view access for the instructors]

## Dataset source and terms

- **Dataset:** DataSEC, Zenodo record `10.5281/zenodo.17033970` (version 3, published Sep. 2, 2025), archive `DATASEC.zip`, listed MD5 `29fa9b8cc84cfa69aa4e5e674e780383`. Link: https://zenodo.org/records/17033970
- **Terms of use:** [TO FILL: license shown on the Zenodo record page]. The paper (Fredianelli et al., Scientific Data 12:1712, 2025, doi 10.1038/s41597-025-05991-w) states that source files were released under Creative Commons licenses.
- The raw archive and model checkpoints are **not** bundled in the submission zip. Download `DATASEC.zip` from the Zenodo record and place it in Google Drive (default path in the notebooks: `MyDrive/ATI G6/DATASEC.zip`; change `ZIP_PATH` or set `DATASEC_ZIP`).
- Note: this copy has 5,048 clips, while the paper's Table 2 lists 4,292. Every class matches the paper except Music (1,001 clips instead of 245).

## Setup

Both notebooks are written for Google Colab with a free T4 GPU, where PyTorch is preinstalled. For a local machine:

```bash
pip install -r code/requirements.txt
```

## Execution order

1. `code/notebooks/Midterm_fixed.ipynb` (Midterm baselines). Set `DATASEC_ZIP` and `OUT_DIR` if your Drive paths differ. It extracts the dataset, builds the fixed split, trains Small CNN, MobileNetV3-Small and CRNN, and writes `config.json`, `split_indices.csv`, `baseline_results.csv`, `per_class_f1_test.csv`, `accuracy_by_duration.csv`, `predictions_clean.csv`, `error_audit.csv`, the Midterm figures and the checkpoints.
2. `code/notebooks/Final_RQ1.ipynb` (RQ1 robustness). Set `DATASEC_ZIP`, `MID_DIR` (the Midterm output folder) and `OUT_DIR`. It reloads the Midterm checkpoints and the fixed split, checks that clean Macro-F1 reproduces the Midterm values (Small CNN 0.7140, MobileNetV3-Small 0.8975, CRNN 0.8363), applies six perturbations at three severities, and writes the Final tables, figures and `predictions_final.csv`.

The notebooks in this folder are the executed versions with their outputs kept.

## Folder content

| Folder | Content |
|---|---|
| `code/notebooks/` | Executed notebooks |
| `code/configs/` | `config.json` (Midterm features, training hyperparameters, normalization statistics, class list), `final_config.json` (perturbation grid and Final settings) |
| `code/splits/` | `split_indices.csv`: train / val / test assignment of every clip (stratified 70/15/15, seed 42; 3,533 / 757 / 758) |
| `code/results/tables/` | CSV result tables |
| `code/results/figures/` | Figures used in the report |
| `code/results/predictions/` | Per-clip probabilities: `predictions_clean.csv` (Midterm), `predictions_final.csv` (all models and perturbation conditions; columns `sample_id, true_class, pred_class, probabilities, duration, model, split, perturbation, severity`) |
| `code/results/environment.md` | Hardware, library versions and seeds |

## Mapping from report to notebooks and files

| Report item | Produced by | File |
|---|---|---|
| Table 1: research questions, hypotheses, controls | written in the report (Section 1.2) | none |
| Table 2: dataset split summary | Midterm notebook | `splits/split_indices.csv` |
| Figure 1: clips per class | Midterm notebook | `results/figures/class_distribution.png` |
| Table 3: training and architecture configuration | Midterm notebook | `results/tables/baseline_results.csv`, `configs/config.json` |
| Table 4: perturbations and severity levels | Final notebook | `configs/final_config.json` |
| Table 5: Midterm baseline results | Midterm notebook | `results/tables/baseline_results.csv` |
| Table 6: Macro-F1 across the 18 perturbed conditions | Final notebook | `results/tables/robustness_results.csv` |
| Figure 2: degradation curves | Final notebook | `results/figures/degradation_curves.png` |
| Table 7: mean and worst-case ΔF1 | Final notebook | `results/tables/delta_f1_summary.csv` |
| Figure 3: per-class degradation at severity 3 | Final notebook | `results/figures/per_class_degradation_sev3.png`, `results/tables/per_class_degradation.csv` |
| Table 8: severity-3 ΔF1 with 95% paired-bootstrap CI | Final notebook | `results/tables/robustness_results.csv` |
| Figure 4: confusion-matrix transitions | Final notebook | `results/figures/confusion_transitions_best_model.png`, `results/tables/confusion_transitions.csv` |
| Table 9: severity-3 failure examples | Final notebook | `results/tables/flip_examples_sev3.csv`, `results/tables/flip_rates.csv` |
| Figure 5: clean versus corrupted spectrograms | Final notebook | `results/figures/flip_examples.png` |
| Acoustic descriptor correlations (Section 6) | Final notebook | `results/tables/class_acoustics.csv`, `results/tables/acoustic_correlations.csv` |
| Tables 10 and 11 (appendices) | written by the group | none |

## Reproducibility notes

- Primary seed: 42 (Python, NumPy, PyTorch, and the stratified split). Perturbation draws are deterministic, seeded per (seed, perturbation, severity, clip index). Paired-bootstrap CIs use 200 resamples of the 758 test clips.
- One training run per model (no multi-seed training). Test labels are never changed and no perturbation parameter was tuned on test results.
