# EIIE 기반 산업 포트폴리오 강화학습 연구 노트

이 저장소는 Zhang et al. (2017) *Deep Reinforcement Learning for Portfolio Management*에서 제안한 EIIE(Ensemble of Identical Independent Evaluators) 아키텍처를 국내 산업 섹터 데이터에 맞게 재현한 실험 코드와 데이터 전처리 파이프라인을 제공합니다. 실험의 핵심은 (1) 산업별 재무·거시 지표를 포함하는 상태 공간을 설계하고, (2) FinRL 포트폴리오 환경에서 EIIE 정책 네트워크를 학습시킨 뒤, (3) 균등 분산 투자 전략(Uniform Buy-And-Hold, 이하 UBAH)과 정량 지표로 비교하는 것입니다.

## 1. 참고 문헌 및 구현 의존성
- **핵심 레퍼런스**  
  Zhang, Z., Zohren, S., & Roberts, S. J. (2017). *Deep Reinforcement Learning for Portfolio Management*.  
  논문에서 제시한 Conv1d+Softmax 구조를 그대로 유지하되, 한국 산업 섹터 데이터에 맞춰 입력 피처를 확장했습니다.
- **구현 프레임워크**  
  [FinRL](https://github.com/AI4Finance-Foundation/FinRL)의 `PortfolioOptimizationEnv`, `DRLAgent`, `EIIE` 클래스를 사용합니다. FinRL 버전에 따라 `time_window`, `features`, `cash_bias` 등의 인자 명이 달라질 수 있으므로 `requirements.txt` 버전 고정이 필수입니다.
- **지표 계산**  
  [QuantStats](https://github.com/ranaroussi/quantstats)를 활용하여 Sharpe, Sortino, MDD, CAGR, Omega 등 위험조정성과 지표를 계산합니다.

## 2. 데이터 전처리 파이프라인 (`전처리.py`)
### 2.1 입력 가정
- 하나의 엑셀 파일에 산업별 재무제표, 업종별 시가총액, 거시지표, ETF 가격 정보가 시트별로 나뉘어 있다고 가정합니다. 기본 시트명은 `산업`, `업종별시가총액`, `경제지표`, `ETF정보`입니다.
- 각 시트에는 `Symbol Name`(산업 코드/이름)와 날짜 열이 존재하며, 일부는 월별/분기별 보고서 형태입니다.

### 2.2 주요 처리 단계
1. **문자 정규화**: 한글 산업 명칭의 접두사 제거, 영문/한글 매핑, 공백 정리.
2. **Melt & Pivot**: 시트를 모두 `(Symbol Name, time, 항목)` 형태로 변환한 뒤 피벗 테이블로 재구성하여 MultiIndex 시계열을 만듭니다.
3. **날짜 정렬 및 보간**: 보고서 기준일을 실거래일에 맞춰 이동시키고, 산업별로 forward-fill을 수행합니다.
4. **로그 변환 및 부호 분리**: 양수 지표는 `np.log`로 스케일링하고, 음수 가능 지표는 `(+/-)` 컬럼을 분리하여 절댓값 로그로 표현합니다.
5. **월간 샘플링**: 학습 안정성을 위해 월초 데이터(`resample('MS')`)만 남깁니다.
6. **산출물**:  
   - `통합데이터.xlsx`: 모든 산업×날짜×피처가 포함된 학습용 원천 데이터  
   - `산업별통합데이터.xlsx`: 월별 샘플링 이후의 입력 데이터

## 3. 강화학습 환경 정식화 (`EIIE 강화학습.py`)
### 3.1 상태 공간 (State)
각 시점 \( t \)에서의 상태 \( s_t \)는 `timewin` 길이의 롤링 윈도우를 적용한 다음 피처 행렬로 구성됩니다.
- **포함 피처**: 종가(`주가`), 시가총액, 총자산, 총부채, 현금흐름, 산업별 경제지표, 주요 원자재/금리 지표 등 약 40개 변수.
- **정규화**: 현재 설정은 무정규화(raw 값)이며, 종가만 첫 값으로 나눠 정규화합니다. 필요 시 `GroupByScaler`(MaxAbs)로 대체 가능합니다.

### 3.2 행동 공간 (Action)
행동 \( a_t \)는 각 산업 섹터에 대한 비중 벡터입니다. Softmax 출력을 사용해 \(\sum_i a_{t,i} = 1\)을 보장합니다. FinRL 환경 옵션에 따라 현금 자산을 추가할 수도 있습니다 (`cash_bias=True`).

### 3.3 보상 함수 (Reward)
기본 보상은 포트폴리오 가치의 로그 수익률입니다.
\[
r_t = \log\left(\frac{V_{t+1}}{V_t}\right)
\]
환경 내부에서 거래 수수료(`fee=0.0012`)가 반영되며, `reward_scaling=0.99995`로 점진적인 감소를 적용해 장기 학습에서의 보상 폭발을 완화합니다.

### 3.4 에피소드 구성
- 학습 구간: `time < 2021-04-01`
- 테스트 구간: `time ≥ 2021-04-01`
- `timewin=15`일을 버퍼로 사용하므로 실사용 기간은 그 이후부터 시작됩니다.

## 4. EIIE 정책 네트워크 구조
FinRL의 `EIIE` 구현은 다음 블록으로 구성됩니다.
1. **Shared 1D CNN**  
   - 입력 텐서 형태: `(batch, feature_dim, timewin, num_assets)`  
   - 동일한 Conv1d 커널이 자산별로 적용되며, 시간 축을 따라 패턴을 추출합니다.
2. **Feature Fusion & Attention**  
   - Conv1d 출력은 자산별 임베딩으로 변환됩니다.  
   - 중간 채널 수(`mid_features=30`)와 최종 채널 수(`final_features=20`)는 Optuna 탐색 대상입니다.
3. **Softmax Head**  
   - 출력은 각 자산별 로그 잇점(logits)으로 구성되며, Softmax 후 포트폴리오 비중이 됩니다.
4. **Residual Connection (옵션)**  
   - Cash 비중을 사용하는 경우, 잔여 비중이 현금으로 할당됩니다.

학습 대상 파라미터는 Conv1d 가중치, attention 결합 계수, Softmax 직전 선형층입니다. 논문과 동일하게 ReLU 활성화와 LayerNorm이 사용됩니다.

## 5. 학습 알고리즘
### 5.1 알고리즘 개요
FinRL의 `DRLAgent`는 Stable-Baselines3를 래핑합니다. 본 스크립트에서는 PPO를 기본으로 사용하지만, A2C/TD3로 교체 가능합니다. PPO의 손실 함수는 다음과 같습니다.
\[
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(
    r_t(\theta)\hat{A}_t,
    \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t
\right)\right]
\]
여기서 \( r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \)이며, \(\hat{A}_t\)는 Generalized Advantage Estimation(GAE)로 계산합니다.

### 5.2 Optuna 기반 하이퍼파라미터 탐색
- 탐색 대상: 학습률, 배치 크기, 클리핑 값, `mid_features`, `final_features`
- 목적 함수: 테스트 구간 MDD를 제약한 Sharpe Ratio 최대화
- 탐색 절차:
  1. 샘플 파라미터 집합을 생성
  2. PPO 에이전트를 학습
  3. 테스트 구간 성과를 계산하여 Optuna trial에 기록
  4. 최고 성과 모델을 `joblib`으로 저장 (필요 시 코드 추가)

### 5.3 학습 의사코드
```python
for trial in study:
    params = sample_hyperparams(trial)
    agent = DRLAgent(env=env_train, model_name="PPO", **params)
    trained_model = agent.train_model(total_timesteps=TIMESTEPS)
    stats = evaluate(trained_model, env_test)
    trial.report(stats["sharpe"], step=0)
```

## 6. 재현 절차
### 6.1 환경 셋업
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```
> PowerShell에서 한글이 깨질 경우 `chcp 65001` 실행 후 작업하세요. 모든 파일은 UTF-8로 저장되어 있습니다.

### 6.2 데이터 준비
1. 원본 엑셀을 프로젝트 루트에 복사
2. `python 전처리.py` 실행
3. 생성된 `통합데이터.xlsx`를 열어 NaN/이상치 여부 확인

### 6.3 학습 실행
1. `EIIE 강화학습.py` 상단의 Colab 전용 `!pip install` 셀은 주석 처리
2. `python "EIIE 강화학습.py"` 실행
3. GPU 사용 시 CUDA 환경이 올바르게 구성되어 있는지 확인 (`torch.cuda.is_available()`)

### 6.4 산출물
- `train_portfolio_data.xlsx`, `test_portfolio_data.xlsx`: 기간별 가치곡선
- `weights_*.xlsx`: 에이전트가 선택한 자산별 비중
- 콘솔 로그: Sharpe, Sortino, MDD, CAGR, Omega, 누적 수익률

## 7. 평가 지표 정리
| 지표 | 정의 | 구현 |
| --- | --- | --- |
| Sharpe | \((R_p - R_f)/\sigma_p\) | `quantstats.stats.sharpe(return_series, rf=0.02, periods=252)` |
| Sortino | \((R_p - R_f)/\sigma_{down}\) | `quantstats.stats.sortino(..., rf=0.02)` |
| MDD | \(\max_{t} \frac{\max_{s \le t} V_s - V_t}{\max_{s \le t} V_s}\) | `quantstats.stats.max_drawdown(nav_series)` |
| CAGR | \((V_T / V_0)^{252/n} - 1\) | `quantstats.stats.cagr(nav_series, periods=252)` |
| Omega | \(\frac{\int_{\tau}^{\infty} (1-F(x))dx}{\int_{-\infty}^{\tau} F(x)dx}\) | `quantstats.stats.omega(return_series, required_return=0.02)` |

## 8. 자주 발생하는 이슈
- **FinRL 버전 차이**: `PortfolioOptimizationEnv` 인자 명이 바뀌면 ImportError가 발생합니다. `requirements.txt` 버전을 유지하세요.
- **엑셀 인코딩 문제**: Windows 엑셀에서 저장 시 UTF-8이 아닌 경우, pandas가 열 이름을 제대로 읽지 못합니다. 가능하면 `UTF-8 (CSV)` 내보내기 후 pandas에서 읽어 다시 저장하세요.
- **Optuna trial 누락**: 실험 중간에 예외가 발생하면 trial이 `FAIL` 처리됩니다. 로그를 남기고 재시도하거나 `catch` 옵션을 지정하세요.
- **GPU/CPU 자원 부족**: `torch.set_num_threads(12)`가 기본값입니다. 로컬 머신 코어 수보다 크면 다운될 수 있으니 적절히 조절하세요.

## 9. 후속 연구 제안
1. **거래 제약조건 추가**: 섹터 비중 상·하한, 거래량 제한 등 현실적인 제약을 `Reward` 가중치로 반영.
2. **동적 위험중립 금리**: QuantStats 계산에 고정 2% 대신 무위험 금리 시계열을 입력.
3. **Factor-aware State**: Fama-French 3/5 팩터, 거시변수 Surprise 지표 등 추가 입력을 실험.
4. **다중 목표 최적화**: Sharpe와 Turnover를 동시 최적화하는 프레임워크 구축 (예: CMDP).

---
질문이나 아이디어가 있다면 GitHub 이슈에 남겨 주세요. 연구 경험을 공유하고 싶으신 분은 언제든 환영합니다. 😊
