# Reinforcement Portfolio Trader (EIIE)

이 저장소는 *개발자 관점*에서 강화학습 기반 산업 포트폴리오 모델을 재현하는 코드 모음입니다. 핵심 포인트는 “데이터를 어떻게 다듬고, 어떤 모델을 돌리고, 어떤 산출물을 만든다”에 초점을 맞춥니다.

---

## 1. 디렉터리 구성 & 주요 스크립트
| 파일 | 역할 |
| --- | --- |
| `전처리.py` | 원천 엑셀(재무/거시/ETF)을 읽어 학습용 시계열을 빌드합니다. |
| `EIIE 강화학습.py` | FinRL의 EIIE 정책 네트워크를 학습시키고 Buy & Hold와 비교합니다. |
| `requirements.txt` | 재현에 필요한 파이썬 패키지 목록입니다. |
| `강화학습 기반 산업 포트폴리오 트레이딩 모델.pdf` | 해당 프로젝트의 결과물(논문). |

> 데이터(.xlsx)는 `.gitignore`에 포함되어 있으니, 개인 데이터는 로컬에만 두세요.

---

## 2. 데이터 파이프라인 (전처리.py)
### 입력 기대치
- 하나의 엑셀 파일에 다음 시트가 존재한다고 가정합니다. (예: `dataset.xlsx`)
  - `산업`: 산업 ETF 가격, 시가총액
  - `업종별시가총액`: 산업별 시총
  - `경제지표`: 금리/원자재/환율 등 거시 지표
  - `ETF정보`: ETF 메타데이터

### 실행 흐름
1. **문자 정리**: 산업 명칭에서 접두사 제거, 한글↔영문 매핑, 공백 제거.
2. **피벗 변환**: 모든 시트를 `(Symbol Name, time, Item Name)` -> Wide 형태로 변환.
3. **재무제표 시차 보정**: 분기 보고서 공개 지연을 고려해 1분기=6월, 2분기=8월, 3분기=12월, 4분기=다음 해 4월의 영업일에 반영.
4. **피처 생성**:
   - ETF 가격 → 시작값 정규화.
    - 나머지 피처 → 로그 변환. 음수인 경우 0으로 대체 후, 부호 반전 파생 변수 생성(양수일 때만 로그).
5. **월간 샘플링**: `resample('MS')`로 월초 데이터만 남깁니다.
6. **산출물**:
   - `통합데이터.xlsx`: 산업×날짜×피처 전체 테이블.
   - `산업별통합데이터.xlsx`: 월간 샘플링 결과, 학습 스크립트 입력용.

### 실행 예시
```powershell
python 전처리.py
```
> 실행 후 동일 폴더에 두 개의 엑셀 파일이 생성됩니다.

---

## 3. 모델 파이프라인 (EIIE 강화학습.py)
### 사용 프레임워크
- [FinRL](https://github.com/AI4Finance-Foundation/FinRL) `PortfolioOptimizationEnv`
- Stable-Baselines3 PPO (`DRLAgent` 래퍼)
- PyTorch 기반 EIIE 아키텍처

### 입력 데이터
- `전처리.py` 결과물인 `산업별통합데이터.xlsx`
- 기간 설정: 학습 2014-04-02 ~ 2021-03-31, 테스트 2021-04-01 ~ 2025-01-31
- 산업 12개, 피처 41개 (시장/재무/거시)

### 주요 하이퍼파라미터
| 파라미터 | 값 | 설명 |
| --- | --- | --- |
| `timewin` | 15 | 15개월 롤링 윈도우 |
| `kernel_size` | 3 | 1D CNN 커널 |
| `mid_features` / `final_features` | 30 / 20 | EIIE 중간/종결 채널 |
| `batch_size` | 128 | PPO 미니배치 |
| `episodes` | 200 | 학습 반복 횟수 |
| `reward_scaling` | 0.99995 | 수익률 스케일링 |
| `fee` | 0.0012 | 거래 비용 |

### 모델 로직 개요
1. **환경 구성**: `PortfolioOptimizationEnv`에 전처리된 데이터, 피처 목록, 수수료, 리워드 스케일링 등을 지정.
2. **EIIE 네트워크**: 모든 자산에 Shared Conv1d → Feature Fusion → Softmax로 비중 산출.
3. **PPO 학습**:  
   \[ L^{CLIP}(\theta) = \mathbb{E}[\min(r_t(\theta)\hat{A}_t, \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)] \]  
   - \( r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \)  
   - \( \hat{A}_t \): GAE Advantage
4. **평가**: 동일 기간에 Buy & Hold 전략을 실행해 비교.

### 실행 예시
```powershell
python "EIIE 강화학습.py"
```
> 실행 시 학습 로그, 평가 지표, 엑셀 산출물(train/test NAV, weights)이 생성됩니다.

---

## 4. 개발 환경 세팅
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```
- PowerShell에서 한글이 깨지면 `chcp 65001` 실행.
- GPU 사용 시 CUDA 드라이버가 설치되어 있어야 합니다 (`torch.cuda.is_available()` 확인).

---

## 5. 학습/평가 산출물
| 파일 | 용도 |
| --- | --- |
| `train_portfolio_data.xlsx` | 학습 구간 NAV |
| `test_portfolio_data.xlsx` | 테스트 구간 NAV |
| `weights_*.xlsx` | 각 시점의 자산 비중 |
| 콘솔 출력 | Sharpe, Omega, MDD, 누적 수익률, CAGR 등 |

성과 지표 계산은 `quantstats`에 맞춰 아래 수식을 사용합니다.
- Sharpe: \( (R_p - R_f) / \sigma_p \), \( R_f = 0.02 \)
- MDD: \( \max_t \frac{\max_{s \le t} V_s - V_t}{\max_{s \le t} V_s} \)
- Omega: \( \frac{\int_\tau^\infty (1-F(x))dx}{\int_{-\infty}^\tau F(x)dx} \), \( \tau = 0.02 \)
- CAGR: \( (V_T / V_0)^{252/n} - 1 \)

샘플 결과 (학습/테스트 구간):
| 구간 | 지표 | EIIE | Buy & Hold |
| --- | --- | --- | --- |
| 학습 | Sharpe | 0.488 | 0.173 |
| 학습 | Omega | 1.092 | 1.033 |
| 학습 | 누적 수익률 | 2.244 | 1.414 |
| 테스트 | Sharpe | 0.162 | -0.042 |
| 테스트 | MDD | -0.175 | -0.312 |
| 테스트 | 누적 수익률 | 1.171 | 1.046 |

---

## 6. 자주 겪는 문제와 대응
- `PortfolioOptimizationEnv` 호출 시 인자 에러 → FinRL 버전 확인(README의 설정과 동일하게 설치).
- 엑셀 컬럼이 깨짐 → 원본 파일을 UTF-8로 재저장하거나 CSV로 변환 후 로딩.
- Trial 도중 중단 → 로그 확인 후 `try/except` 또는 Optuna `catch` 옵션 사용.
- CPU 코어 부족 → `torch.set_num_threads()` 값을 환경에 맞게 조정.

---

## 7. 다음 단계 아이디어
- 비중 상한/하한, 거래 제한 등 현실 제약 추가.
- 무위험 금리를 동적으로 반영(금리 시계열 입력).
- Fama-French 팩터, Macro Surprise 등 요인 기반 피처 증설.
- Sharpe + Turnover 등 다중 목표를 동시에 최적화하는 CMDP 실험.

---
버그나 개선 아이디어가 있다면 GitHub 이슈로 남겨 주세요. 😄
