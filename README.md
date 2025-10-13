# EIIE 기반 산업 포트폴리오 강화학습 연구 노트

국내 산업 섹터 데이터를 이용해 Zhang et al. (2017)의 EIIE(Ensemble of Identical Independent Evaluators) 아키텍처를 재현하고, 균등 분산 투자 전략(Uniform Buy-And-Hold, UBAH)과 비교하는 실험 코드 모음입니다.  
핵심 목표는 다음 세 가지입니다.
1. 산업별 재무·거시 지표를 포함하는 상태 공간 정의
2. FinRL 포트폴리오 환경에서 EIIE 정책 네트워크 학습
3. Sharpe, Sortino, MDD, CAGR, Omega 등 정량 지표로 성능 비교

---

## 1. 참고 문헌 및 구현 의존성
- **핵심 레퍼런스**  
  Zhang, Z., Zohren, S., & Roberts, S. J. (2017). *Deep Reinforcement Learning for Portfolio Management*.  
  논문에서 제안한 Conv1d + Softmax 구조를 유지하면서 한국 산업 데이터를 입력으로 확장했습니다.
- **구현 프레임워크**  
  [FinRL](https://github.com/AI4Finance-Foundation/FinRL)의 `PortfolioOptimizationEnv`, `DRLAgent`, `EIIE` 모듈을 사용합니다. 버전마다 인자 이름이 달라질 수 있으므로 `requirements.txt`의 버전을 그대로 유지하세요.
- **성과 지표 라이브러리**  
  [QuantStats](https://github.com/ranaroussi/quantstats)를 이용하여 Sharpe, Sortino, MDD, CAGR, Omega 지표를 계산합니다.

---

## 2. 데이터 전처리 파이프라인 (`전처리.py`)
### 2.1 입력 가정
- 하나의 엑셀 파일에 산업별 재무제표, 업종별 시가총액, 거시지표, ETF 정보가 시트별로 구분되어 있다고 가정합니다. 기본 시트명은 `산업`, `업종별시가총액`, `경제지표`, `ETF정보`입니다.
- 각 시트에는 `Symbol Name`(산업 코드/명)과 날짜 열이 존재하며 월별 또는 분기별 보고 주기를 가집니다.

### 2.2 주요 처리 단계
1. **문자 정규화**: 산업 명칭의 접두사/공백 제거, 한글-영문 매핑.
2. **Melt & Pivot**: 시트를 모두 `(Symbol Name, time, 항목)` 형태로 만든 뒤 피벗 테이블로 재구성하여 MultiIndex 시계열 생성.
3. **날짜 정렬 및 보간**: 보고서 기준일을 실제 거래일에 맞게 이동시키고, 산업별로 forward-fill 수행.
4. **로그 변환 및 부호 분리**: 양수 지표는 `np.log`로 스케일링하고, 음수 가능 지표는 `(+/-)` 컬럼으로 분리 후 절댓값 로그 변환.
5. **월간 샘플링**: 학습 안정성을 위해 `resample('MS')`로 월초 데이터만 남깁니다.
6. **출력물**:
   - `통합데이터.xlsx`: 산업 × 날짜 × 피처 전체가 포함된 원천 데이터.
   - `산업별통합데이터.xlsx`: 월 샘플링 결과로 학습 스크립트에서 직접 사용합니다.

---

## 3. 강화학습 정식화 (`EIIE 강화학습.py`)
### 3.1 상태 공간
각 시점 $t$에서의 상태 $s_t$는 `timewin` 길이의 롤링 윈도우를 적용한 피처 행렬입니다.
- 포함 피처: 종가, 시가총액, 자산/부채, 현금흐름, 산업별 거시 변수, 원자재 및 금리 지표 등 약 40개.
- 정규화: 종가는 첫 값으로 나누어 정규화하고, 나머지는 원시 값(raw)을 사용합니다. 필요 시 `GroupByScaler`(MaxAbs)로 교체 가능합니다.

### 3.2 행동 공간
행동 $a_t$는 산업 섹터별 비중 벡터입니다. Softmax 출력을 사용하여 $ \sum_i a_{t,i} = 1 $을 보장하며, `cash_bias=True`를 설정하면 현금 자산 비중을 추가할 수 있습니다.

### 3.3 보상 함수
기본 보상은 포트폴리오 가치의 로그 수익률입니다.
$$
r_t = \log\left(\frac{V_{t+1}}{V_t}\right)
$$
환경 내부에서 거래 수수료(`fee=0.0012`)가 반영되고, `reward_scaling=0.99995`를 적용해 장기 학습 시 보상 폭발을 완화합니다.

### 3.4 에피소드 구성
- 학습(train) 구간: `time < 2021-04-01`
- 테스트(test) 구간: `time ≥ 2021-04-01`
- `timewin=15` 데이터를 버퍼로 사용하므로 실제 학습에 활용되는 기간은 `timewin` 이후부터 시작됩니다.

---

## 4. EIIE 정책 네트워크 구조
FinRL의 `EIIE` 아키텍처는 다음 블록으로 구성됩니다.
1. **Shared 1D CNN**  
   - 입력 텐서: `(batch, feature_dim, timewin, num_assets)`  
   - 동일한 Conv1d 커널을 자산별로 공유하여 시간 패턴을 추출합니다.
2. **Feature Fusion & Attention**  
   - Conv1d 출력은 섹터별 임베딩으로 변환되며, `mid_features=30`, `final_features=20`은 Optuna가 탐색합니다.
3. **Softmax Head**  
   - 출력은 각 섹터의 로그 이점(logit)이며 Softmax를 거쳐 비중이 됩니다.
4. **Residual / Cash Node**  
   - 현금 비중을 사용할 경우 잔여 비중이 자동으로 현금에 할당됩니다.

학습 대상 파라미터는 Conv1d 가중치, attention 결합 계수, Softmax 직전 선형층이며, 논문과 동일하게 ReLU 활성화와 LayerNorm이 사용됩니다.

---

## 5. 학습 알고리즘
### 5.1 PPO 손실 함수
FinRL의 `DRLAgent`는 Stable-Baselines3를 래핑하며 기본적으로 PPO를 사용합니다.

$$
L^{\text{CLIP}}(\theta)
= \mathbb{E}_t \left[
  \min\left(
    r_t(\theta)\,\hat{A}_t,\,
    \mathrm{clip}\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right)\hat{A}_t
  \right)
\right]
$$

여기서 $r_t(\theta) = \dfrac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$, $\hat{A}_t$는 Generalized Advantage Estimation(GAE)로 계산합니다.

### 5.2 Optuna 기반 하이퍼파라미터 탐색
- 탐색 대상: 학습률, 배치 크기, 클리핑 값, `mid_features`, `final_features`
- 목적 함수: 테스트 구간 MDD를 제약한 Sharpe Ratio 최대화
- 절차:
  1. Trial마다 하이퍼파라미터 샘플링
  2. PPO 학습 및 모델 저장
  3. 테스트 환경에서 성능 측정 후 Trial에 보고

### 5.3 학습 의사코드
```python
for trial in study:
    params = sample_hyperparams(trial)
    agent = DRLAgent(env=env_train, model_name="PPO", **params)
    model = agent.train_model(total_timesteps=TIMESTEPS)
    stats = evaluate(model, env_test)
    trial.report(stats["sharpe"], step=0)
```

---

## 6. 재현 절차
### 6.1 환경 설정
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```
> PowerShell에서 한글이 깨질 경우 `chcp 65001`을 먼저 실행하세요. 모든 파일은 UTF-8 인코딩으로 저장되어 있습니다.

### 6.2 데이터 준비
1. 원본 엑셀을 프로젝트 루트에 복사
2. `python 전처리.py` 실행
3. 생성된 `통합데이터.xlsx`에서 NaN/이상치를 눈으로 확인

### 6.3 학습 실행
1. `EIIE 강화학습.py` 최상단의 Colab 전용 `!pip install` 줄을 주석 처리
2. `python "EIIE 강화학습.py"` 실행
3. GPU 사용 시 `torch.cuda.is_available()`를 통해 CUDA 인식 여부 확인

### 6.4 산출물
- `train_portfolio_data.xlsx`, `test_portfolio_data.xlsx`: 기간별 가치곡선
- `weights_*.xlsx`: 학습된 정책이 선택한 섹터 비중
- 콘솔 로그: Sharpe, Sortino, MDD, CAGR, Omega, 누적 수익률

---

## 7. 성과 지표 정의
| 지표 | 수식 | 구현 |
| --- | --- | --- |
| Sharpe | $(R_p - R_f) / \sigma_p$ | `quantstats.stats.sharpe(return_series, rf=0.02, periods=252)` |
| Sortino | $(R_p - R_f) / \sigma_{down}$ | `quantstats.stats.sortino(..., rf=0.02)` |
| MDD | $\max_t \frac{\max_{s \le t} V_s - V_t}{\max_{s \le t} V_s}$ | `quantstats.stats.max_drawdown(nav_series)` |
| CAGR | $(V_T / V_0)^{252/n} - 1$ | `quantstats.stats.cagr(nav_series, periods=252)` |
| Omega | $\frac{\int_{\tau}^{\infty} (1 - F(x))\,dx}{\int_{-\infty}^{\tau} F(x)\,dx}$ | `quantstats.stats.omega(return_series, required_return=0.02)` |

---

## 8. 자주 발생하는 이슈
- **FinRL 버전 차이**: `PortfolioOptimizationEnv` 인자 이름이 바뀌면 ImportError가 발생합니다. `requirements.txt` 버전을 유지하세요.
- **엑셀 인코딩 문제**: Windows Excel에서 UTF-8이 아닌 형식으로 저장하면 pandas가 컬럼 이름을 깨뜨립니다. 필요 시 CSV(UTF-8)로 내보낸 뒤 다시 엑셀로 저장하세요.
- **Optuna Trial 실패**: 학습 중 예외가 발생하면 Trial이 `FAIL` 처리됩니다. 로그를 확인하고 `catch` 옵션을 설정해 자동 재시도할 수 있습니다.
- **자원 부족**: `torch.set_num_threads(12)`가 기본값입니다. 코어 수가 적다면 값을 줄이세요.

---

## 9. 후속 연구 제안
1. **거래 제약조건 추가**: 섹터 비중 상·하한, 거래량 제한 등을 Reward 또는 Constraint로 반영.
2. **동적 무위험 금리**: QuantStats 계산에 고정 2% 대신 실제 무위험 금리 시계열을 입력.
3. **Factor-aware 상태 설계**: Fama-French 팩터, Macro Surprise 지표 등을 추가로 입력하여 일반화 성능 비교.
4. **다중 목표 최적화**: Sharpe와 Turnover를 동시에 고려하는 CMDP 형태의 학습 실험.

---
문의나 아이디어가 있다면 GitHub 이슈에 자유롭게 남겨 주세요. 😊
