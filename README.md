# 금융 최적화 RL (EIIE)

국내 산업 섹터 데이터를 활용해 EIIE(Ensemble of Identical Independent Evaluators) 구조의 강화학습 포트폴리오 모델을 학습·평가하는 프로젝트입니다. 사용자 정의 전처리, FinRL 환경 래퍼, 학습 이후 성과 지표 분석을 통해 균등 분산 투자(UBAH) 전략과 성과를 비교합니다.

## 저장소 구성
- `전처리.py` – 원본 엑셀 데이터를 정리하고 경제 지표를 정합한 뒤 `통합데이터.xlsx`, `산업별통합데이터.xlsx`를 생성합니다.
- `EIIE 강화학습.py` – EIIE 에이전트 학습 및 평가 파이프라인으로 Optuna 기반 하이퍼파라미터 탐색과 QuantStats 성과 리포트를 포함합니다.
- `산업포트폴리오 트레이딩 모델.pdf` – 프로젝트 개요 프레젠테이션 자료입니다.

## 환경 설정
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

최신 FinRL 구현을 활용하려면 다음처럼 클론 후 설치하세요.
```powershell
git clone https://github.com/AI4Finance-Foundation/FinRL.git
pip install -e FinRL
```

> **인코딩 안내**  
> 모든 소스 파일과 README는 UTF-8 인코딩으로 저장되어 있으며, GitHub 웹에서 정상적으로 렌더링됩니다. 에디터에서도 UTF-8로 열고 수정하세요.

## 데이터 준비 절차
1. 산업별 재무제표와 거시지표가 포함된 엑셀 원본 파일을 스크립트와 같은 폴더에 둡니다. 전처리 스크립트는 `산업`, `업종별시가총액`, `경제지표`, `ETF정보` 등 한글 시트명을 기대합니다.
2. `python 전처리.py`를 실행해 `통합데이터.xlsx`, `산업별통합데이터.xlsx`를 생성합니다.
3. 학습 전에 생성된 엑셀 파일의 날짜 범위와 컬럼 매핑이 원하는 형태인지 확인합니다.

## 학습 및 평가
- 본 스크립트는 원래 Colab/Jupyter 노트북을 기반으로 작성되어 앞부분에 `!pip install …` 셀 구문이 포함되어 있습니다. 독립 실행 시엔 해당 줄을 주석 처리하고 `requirements.txt` 설치만 진행하세요.
- 학습 실행:
  ```powershell
  python "EIIE 강화학습.py"
  ```
- 결과물로 Optuna 스터디 로그, Sharpe·Sortino·MDD·Omega·CAGR 등 QuantStats 지표, 학습/테스트 구간의 자산 곡선 및 자산 배분 엑셀 파일이 생성됩니다.

## 향후 작업 제안
- 하드코딩된 하이퍼파라미터를 CLI 옵션 또는 설정 파일로 분리해 재현성을 개선하세요.
- 전처리 파이프라인과 환경 리셋이 정상 동작하는지 확인하는 유닛 테스트/스모크 테스트를 추가하세요.
- 공개 저장소로 유지하려면 민감한 엑셀 데이터를 제외하거나 별도 배포 절차를 문서화하세요.
