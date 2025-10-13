# EIIE 기반 산업 포트폴리오 강화학습

고정관념 그대로 돌리기보다는, 논문을 곁에 두고 한 단계씩 따라 할 수 있도록 정리한 프로젝트 안내입니다. 이 저장소는 한국 산업 섹터 데이터를 활용해 EIIE(Ensemble of Identical Independent Evaluators) 구조의 강화학습 포트폴리오 에이전트를 학습시키고, 균등 분산 투자(Uniform Buy And Hold, UBAH) 전략과 비교하는 과정을 담고 있습니다.

## 프로젝트 한눈에 보기
- 산업별 재무제표 + 거시지표 엑셀을 `전처리.py`로 묶어 학습용 시계열을 만듭니다.
- FinRL에 포함된 `PortfolioOptimizationEnv`와 `EIIE` 아키텍처로 강화학습 환경을 구성합니다.
- Optuna로 하이퍼파라미터를 탐색하고, QuantStats로 Sharpe, Sortino, MDD, CAGR, Omega 등을 계산합니다.
- 결과는 학습/검증 구간별 자산 곡선, 포트폴리오 비중 엑셀, 지표 요약으로 남습니다.

## 참고 & 이론 배경
| 구분 | 자료 | 메모 |
| --- | --- | --- |
| 원천 모델 | Zhang et al., 2017, “Deep Reinforcement Learning for Portfolio Management” | EIIE가 제안된 논문. 동일한 Conv1d 블록을 종목별로 적용해 weight vector를 만들고, Softmax로 비중을 정규화합니다. |
| 구현 참고 | AI4Finance FinRL, Portfolio Optimization 모듈 | 본 저장소 스크립트가 직접 활용하는 라이브러리. 최신 버전에서는 API 시그니처가 조금씩 바뀌니 requirements 버전에 주의하세요. |
| 금융 지표 | QuantStats documentation | Sharpe, Sortino, Omega 계산 시 입력값(위험중립 금리, 기간)에 따라 수치가 크게 바뀌므로 필수 확인. |

## 파일 구성과 역할
- `전처리.py`  
  - 기대 입력: 한국 산업 섹터별 재무제표/거시지표 엑셀 (`산업`, `업종별시가총액`, `경제지표`, `ETF정보` 등의 시트명).  
  - 주요 처리: 문자열 정리(띄어쓰기 제거, 한글-영문 매핑), 멀티인덱스 → 피벗 변환, 월간 샘플링, NaN 보간, 로그 스케일링.  
  - 산출물: `통합데이터.xlsx`, `산업별통합데이터.xlsx` (학습 스크립트 입력).
- `EIIE 강화학습.py`  
  - 학습/평가 전체 파이프라인.  
  - 핵심 하이퍼파라미터: `timewin=15`, `fee=0.0012`, `reward_scaling=0.99995`, `mid_features=30`, `final_features=20`.  
  - Optuna 탐색 후, 균등분산(UBAH)과 EIIE를 비교하고, 결과 엑셀/지표를 저장.
- `산업포트폴리오 트레이딩 모델.pdf`  
  - 프로젝트 요약 슬라이드.
- `requirements.txt`, `.gitignore`, `README.md`  
  - 재현 환경과 저장소 설명.

## 환경 준비 가이드 (PowerShell 기준)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

> 모든 스크립트와 README는 UTF-8 인코딩입니다. PowerShell이 한글을 깨뜨릴 때는 `chcp 65001`을 먼저 실행하거나, VS Code에서 파일 인코딩을 UTF-8로 강제 지정하세요. Git에서 CRLF 경고가 보이면 `git config core.autocrlf true` 설정도 고려하세요.

### FinRL 최신 버전 사용 시
```powershell
git clone https://github.com/AI4Finance-Foundation/FinRL.git
pip install -e .\FinRL
```
만약 `PortfolioOptimizationEnv`의 인자가 바뀌었다면, 스크립트 내 `time_window`, `features`, `valuation_feature` 등의 인자를 맞춰 수정해야 합니다.

## 데이터 준비 절차
1. **원본 엑셀 정리**  
   - 필수 컬럼: `Symbol Name`, `time`, 종가(`시가총액`, `주가` 등), 재무 지표, 거시 변수.  
   - 날짜 열은 `datetime64[ns]`로 변환되도록 날짜 형식을 통일합니다. 엑셀의 병합된 셀은 미리 해체하는 편이 안전합니다.
2. **전처리 스크립트 실행**  
   ```powershell
   python 전처리.py
   ```
   - 생성된 `통합데이터.xlsx`에서 NaN이 예상 범위를 넘으면, 전처리 구간(예: ffill, 로그 변환)을 조정하세요.
3. **학습/검증 구간 확인**  
   - `EIIE 강화학습.py`는 기본적으로 `sptdate = "2021-04-01"`을 기준으로 훈련/테스트를 나눕니다. 필요하면 해당 값을 수정하세요.

## 학습 순서 (Step-by-step)
1. 가상환경 활성화 후 FinRL, QuantStats 등이 정상 설치됐는지 `python -c "import finrl, quantstats"`로 확인.
2. `EIIE 강화학습.py` 최상단의 Colab 전용 `!pip install` 구문은 주석 처리하거나 삭제합니다.
3. `python "EIIE 강화학습.py"` 실행.  
   - `torch.set_num_threads(12)`가 설정되어 있으므로 CPU 코어가 부족한 환경에서는 값을 줄여도 됩니다.  
   - GPU 사용 시 `device = 'cuda:0' if torch.cuda.is_available() else 'cpu'`가 자동 감지합니다.
4. 실행이 완료되면 아래 산출물을 확인합니다.
   - `train_portfolio_data.xlsx`, `test_portfolio_data.xlsx`: 기간별 자산 곡선.
   - `weights` 관련 엑셀/CSV: 종목별 자산 비중.
   - 콘솔 출력: Sharpe/Sortino/MDD/CAGR/Omega 지표 비교.
5. Optuna 탐색 결과를 재사용하려면 `study` 객체를 `joblib.dump`로 저장하는 부분을 추가하세요.

## 핵심 하이퍼파라미터 설명
| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `timewin` | 15 | Conv1d 윈도우 크기. 15거래일 히스토리를 한 번에 입력합니다. |
| `fee` | 0.0012 | 거래 수수료 비율. 왕복 수수료를 모델에 반영합니다. |
| `reward_scaling` | 0.99995 | 보상 스케일링 팩터. 장기간 학습 시 보상 폭발을 완화합니다. |
| `mid_features` | 30 | EIIE 내부 중간 채널 수. Conv1d에서 추출하는 특징 수를 결정합니다. |
| `final_features` | 20 | Softmax 이전 출력 채널 수. |
| `rf` | 0.02 | QuantStats 위험중립 금리. Sharpe/Sortino 계산에 사용합니다. |
| `periods` | 252 | 연환산을 위한 기간 수. 한국 주식 시장 기준으로 252거래일을 사용합니다. |

## 모델 구조 요약
- **상태 표현**: `features` 리스트에 포함된 재무/거시 변수 + 종가를 정규화 없이 그대로 입력. 종목별로 `timewin` 길이의 컨볼루션을 적용합니다.
- **EIIE 아키텍처**  
  1. 입력 텐서: `(batch, timewin, 종목 수, feature 수)`  
  2. 동일한 1D CNN 블록을 종목별로 공유해 시간 패턴을 학습.  
  3. Attention-like 조합층을 거친 뒤 Softmax로 포트폴리오 비중 산출.  
  4. Cash 비중 노드를 추가해 무위험 자산 보유를 허용할 수 있습니다(필요 시 `cash_bias` 옵션 참고).
- **행동 출력**: 자산 비중 벡터. 환경에서 거래 수수료와 슬리피지를 고려해 다음 스텝의 자산 가치로 변환합니다.

## 평가 지표 & 리포팅
- **Sharpe / Sortino**: `quantstats.stats` 모듈 사용, `rf=0.02`, `periods=252`로 연환산.  
- **MDD (최대낙폭)**: `quantstats.stats.max_drawdown`으로 계산.  
- **CAGR**: `quantstats.stats.cagr`. 데이터 기간과 `periods`가 유효한지 점검하세요.  
- **Omega**: `required_return=0.02`로 손실 확률 대비 수익 확률을 비교합니다.  
- **엑셀 리포트**: `train_portfolio_data.xlsx`, `test_portfolio_data.xlsx`, `weights` 엑셀로 후처리/시각화가 가능. Tableau나 Power BI로 연결해 대시보드화하는 것도 좋습니다.

## 실험 팁
- Optuna 탐색 시 탐색 공간을 명시적으로 제한하세요(예: `fee` 0.0005~0.002, `timewin` 10~30). 기본 스크립트는 탐색 공간이 고정값으로 되어 있으니 확장해도 좋습니다.
- 데이터가 길수록 첫 15일(`timewin`)은 모델이 학습에 사용하되 평가에선 잘려나가므로, 충분한 시작 구간을 확보하세요.
- 로그 변환(`np.log`)이 적용된 피처는 0 이하일 때 0으로 덮어씁니다. 재무지표가 음수가 될 수 있는 항목은 `(-)` 파생 컬럼으로 분리 후 절댓값 변환을 수행합니다. 필요하다면 Z-score나 RobustScaler로 대체할 수 있습니다.
- 결과 비교 시 동일 기간의 벤치마크(예: KOSPI 지수)를 함께 계산하면 설득력이 올라갑니다.

## 자주 하는 실수 체크리스트
- ✅ 전처리 결과 엑셀을 학습 디렉터리에 두었나요? (없으면 `FileNotFoundError` 발생)  
- ✅ FinRL 버전이 너무 최신이라 API가 바뀌지 않았나요? (`PortfolioOptimizationEnv` 인자 mismatch)  
- ✅ `device` 설정이 CPU/GPU에 맞게 잡혔나요? (특히 CUDA 설치 여부)  
- ✅ QuantStats가 설치되어 있나요? (`pip install quantstats` 누락 시 ImportError)  
- ✅ Git에 민감한 데이터(원본 엑셀)가 올라가지 않도록 `.gitignore`가 적용됐나요? (기본으로 `*.xlsx`를 무시하지만, 샘플 데이터를 올릴 땐 예외 처리 필요)

## 다음 단계 제안
1. 학습 결과를 노트북(예: Jupyter)으로 시각화하는 리포트를 추가해 논문 스타일의 그림을 만드세요.
2. `argparse`를 도입해 기간(`sptdate`), 수수료(`fee`), `timewin` 등을 커맨드라인에서 바로 바꿀 수 있게 만드세요.
3. 리밸런싱 주기(일간, 주간)나 거래 제약조건(비중 상한/하한)을 다양하게 시험해 후속 연구를 진행하세요.
4. 백테스트 결과를 FactSet, FnGuide 등 외부 데이터와 비교해 데이터 품질을 검증하세요.

---
문의나 공유하고 싶은 팁이 있다면 GitHub 이슈를 열어 주세요. 한글로 편하게 남기셔도 됩니다! 💬
