# Finance Optimization RL (EIIE)

Portfolio optimisation research project that fine‑tunes and evaluates the EIIE (Ensemble of Identical Independent Evaluators) reinforcement learning architecture on Korean industrial sector data. The workflow combines custom preprocessing, FinRL environment wrappers, and post‑training analytics to compare against a uniform buy‑and-hold baseline.

## Repository Layout
- `전처리.py` – cleans the raw Excel workbooks, aligns economic indicators, and exports `통합데이터.xlsx` / `산업별통합데이터.xlsx` that are consumed by the training script.
- `EIIE 강화학습.py` – main training / evaluation pipeline for the EIIE agent, including Optuna hyper‑parameter search and QuantStats reporting.
- `산업포트폴리오 트레이딩 모델.pdf` – slide deck summarising the project.

## Environment Setup
```bash
python -m venv .venv
.venv\Scripts\activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

The script relies on `finrl` (which in turn installs its own dependencies). If you prefer the bleeding-edge implementation, clone `FinRL` and install it in editable mode:
```bash
git clone https://github.com/AI4Finance-Foundation/FinRL.git
pip install -e FinRL
```

## Data Preparation
1. Place the source Excel workbook(s) that contain the sector fundamentals and macro indicators alongside the scripts (the preprocessing code expects Korean sheet names such as `산업`, `업종별시가총액`, `경제지표`, `ETF정보`).
2. Run `python 전처리.py` to generate `통합데이터.xlsx` and `산업별통합데이터.xlsx`.
3. Review the exported files to ensure the date ranges and column mappings meet your requirements before training.

## Training & Evaluation
- The training script was originally authored for a Jupyter/Colab environment (hence the leading `!pip install …` cells). When running it as a standalone module, comment those out and rely on `requirements.txt` instead.
- Launch training with:
  ```bash
  python "EIIE 강화학습.py"
  ```
- Outputs include Optuna study results, QuantStats metrics (Sharpe, Sortino, MDD, Omega, CAGR), and Excel exports for the train/test equity curves and portfolio weights.

## Next Steps
- Replace hard-coded hyper-parameters with CLI arguments or a config file for reproducible experiments.
- Add unit tests or smoke tests that validate data preprocessing and environment resets before running long training sessions.
- Document the exact versions of the raw datasets if you plan to share the repository publicly (sensitive financial data should remain private).
