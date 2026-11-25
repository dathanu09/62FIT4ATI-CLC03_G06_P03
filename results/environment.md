# Environment & Reproducibility Metadata (Group 06 · P03 DataSEC)

## Hardware
- Platform: Google Colab
- GPU: NVIDIA Tesla T4 (CUDA enabled)

## Software Versions (from `config.json` and `final_config.json`)
- Python: `3.13.15` (Midterm run) / `3.13.16` (Final run)
- PyTorch (`torch`): `2.11.0+cu130` (Midterm) / `2.14.1+cu130` (Final)
- `torchaudio`: `2.11.0+cu130`
- `torchvision`: `0.26.0+cu130` (Midterm)
- `librosa`: `0.11.0`
- `soundfile`: `0.14.0`
- `scikit-learn`: `1.6.1`
- `scipy`: `1.16.3`
- `numpy`: `2.1.3` (Midterm) / `2.4.6` (Final)
- `pandas`: `2.2.3`

## Random Seeds & Split
- Primary seed: `42` (set for Python `random`, NumPy, PyTorch CPU/CUDA, and stratified train/val/test split `70% / 15% / 15%` -> `3,533 / 757 / 758` clips)
- Perturbation draws: deterministic generator seeded per `(seed, perturbation, severity, clip_index)`
- Paired bootstrap confidence intervals: `200` resamples over the `758` test clips
