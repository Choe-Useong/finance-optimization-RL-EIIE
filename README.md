# 강화학습 기반 산업 포트폴리오 트레이딩 모델

국내 산업 섹터 ETF를 대상으로 FinRL의 EIIE(Ensemble of Identical Independent Evaluators) 정책 네트워크를 학습시켜 Buy & Hold 전략과 성과를 비교한 연구 코드입니다. 데이터 전처리부터 학습/평가, 결과 해석까지 PDF 보고서의 핵심 내용을 README로 요약했습니다.

## 1. 데이터 구성과 전처리
- **범위**: 2014-04-02 ~ 2025-01-31, 월간 샘플.
- **산업(12)**: 반도체, 방송·통신, 보험, 에너지·화학, 운송, 은행, 자동차, 증권, 철강, 헬스케어, IT 등.
- **피처(41)**  
  - 시장 지표: ETF 수정주가, 시가총액.  
  - 재무제표: 유동/비유동 자산·부채, 매출액, 영업이익, 당기순이익, 총포괄이익 등.  
  - 거시 지표: 금리(미국 국채 10/1년, CD, 국고 10년, 회사채 AA-), 원자재(금, 니켈, 소맥, 전기동, 두바이유), 환율(달러·엔·위안).
- **전처리 요약**
  1. ETF 가격은 시작값으로 정규화.
  2. 나머지 피처는 로그 변환을 기본으로 하되, 음수 값은 0으로 대체하고 부호를 반전한 파생 변수도 생성 후 로그 적용.
  3. 재무제표 공시 시차를 반영: 1분기=6월 첫 영업일, 2분기=8월 첫 영업일, 3분기=12월 두 번째 영업일, 4분기=다음 해 4월 첫 영업일에 반영.
  4. `resample('MS')`로 월초 데이터만 유지.
  5. 최종 산출물: `통합데이터.xlsx`, `산업별통합데이터.xlsx`.

## 2. 모델 구조 (EIIE)
- **입력**: `timewin=15` 길이의 시계열 윈도우와 41개 피처.
- **Shared 1D CNN**: 모든 자산에 동일한 Conv1d 커널 적용.
- **Feature Fusion**: 중간 피처 30채널, 최종 피처 20채널로 압축.
- **Softmax Head**: 자산별 비중을 산출하고 합계 1을 보장. 필요 시 `cash_bias=True`로 현금 노드 추가.
- **보상**: 로그 수익률  
  \[
  r_t = \log\left(\frac{V_{t+1}}{V_t}\right)
  \]
  거래비용 0.0012, 보상 스케일 0.99995 적용.
- **학습 알고리즘**: PPO (Stable-Baselines3)  
  \[
  L^{\text{CLIP}}(\theta)=\mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\right)\right]
  \]
  여기서 \( r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \), \( \hat{A}_t \)는 GAE로 계산.

## 3. 학습 설정
| 파라미터 | 값 |
| --- | --- |
| Time window | 15 |
| 입력 피처 수 | 41 |
| 중간/최종 피처 | 30 / 20 |
| 커널 크기 | 3 |
| 배치 크기 | 128 |
| 에피소드 | 200 |
| 보상 스케일 | 0.99995 |
| 거래비용 | 0.0012 |

- **데이터 분할**: 학습 2014-04-02 ~ 2021-03-31, 테스트 2021-04-01 ~ 2025-01-31.
- **비교 전략**: 동일 비중 Buy & Hold.

## 4. 성과 지표
- Sharpe: \( (R_p - R_f) / \sigma_p \) (무위험 수익률 2% 가정)
- Sortino: \( (R_p - R_f) / \sigma_{\text{down}} \)
- MDD: \( \max_t \frac{\max_{s \le t} V_s - V_t}{\max_{s \le t} V_s} \)
- CAGR: \( (V_T / V_0)^{252/n} - 1 \)
- Omega: \( \frac{\int_{\tau}^{\infty}(1-F(x))dx}{\int_{-\infty}^{\tau}F(x)dx} \) (Threshold \( \tau=2\% \))

QuantStats를 사용해 모든 지표를 계산합니다.

## 5. 실험 결과 (PDF 표 3 요약)
| 구간 | 지표 | EIIE | Buy & Hold |
| --- | --- | --- | --- |
| 학습 | Sharpe | 0.488 | 0.173 |
| 학습 | Omega | 1.092 | 1.033 |
| 학습 | MDD | -0.535 | -0.534 |
| 학습 | 누적 수익률 | 2.244 | 1.414 |
| 학습 | CAGR | 0.083 | 0.035 |
| 테스트 | Sharpe | 0.162 | -0.042 |
| 테스트 | Omega | 1.030 | 0.992 |
| 테스트 | MDD | -0.175 | -0.312 |
| 테스트 | 누적 수익률 | 1.171 | 1.046 |
| 테스트 | CAGR | 0.029 | 0.008 |

- 학습·테스트 모두에서 EIIE가 누적 수익률과 CAGR에서 우위.
- 테스트 구간(약세장)에서도 Sharpe과 Omega가 양수이며, MDD가 낮아 위험 대비 수익이 개선됨.

## 6. 재현 절차
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```
1. 원본 엑셀을 프로젝트 루트에 배치.
2. `python 전처리.py` 실행 → `통합데이터.xlsx`, `산업별통합데이터.xlsx` 생성.
3. `EIIE 강화학습.py` 상단의 Colab 전용 `!pip install` 줄을 비활성화.
4. `python "EIIE 강화학습.py"` 실행.
5. 출력물 확인: `train_portfolio_data.xlsx`, `test_portfolio_data.xlsx`, `weights_*.xlsx`, 콘솔 지표 로그.

> PowerShell에서 한글이 깨질 경우 `chcp 65001`을 먼저 실행하세요. 모든 파일은 UTF-8 인코딩입니다.

## 7. 자주 발생하는 이슈
- **FinRL 버전 변경**: `PortfolioOptimizationEnv` 인자명이 달라질 수 있으므로 `requirements.txt` 버전을 유지하거나 코드 상 인자를 확인.
- **엑셀 인코딩**: UTF-8이 아닌 상태로 저장하면 pandas가 컬럼명을 깨뜨릴 수 있습니다. 필요 시 CSV(UTF-8)로 변환 후 다시 쓰세요.
- **Optuna Trial 실패**: 학습 중 예외로 Trial이 `FAIL`이 되면 로그를 확인하고 `catch` 옵션으로 재시도를 허용하세요.
- **자원 부족**: `torch.set_num_threads(12)`는 CPU 코어가 적을 때 값을 줄이는 것이 안전합니다.

## 8. 후속 과제
1. 섹터 비중 상·하한, 거래량 제약 등을 포함한 현실적 제약 조건 도입.
2. 무위험 금리를 고정값이 아닌 시계열로 반영.
3. Fama-French 팩터나 Macro Surprise 지표 등 추가 피처 주입.
4. Sharpe와 Turnover를 동시에 고려하는 다중 목적(CMDP) 학습 실험.

---
궁금한 점이나 개선 아이디어가 있다면 GitHub 이슈로 공유해 주세요! 😊
