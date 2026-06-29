# F5 실행 문제 점검 리포트

작성일: 2026-06-29

## 결론

F5 실행이 안 되던 핵심 원인은 한 가지가 아니라 여러 설정/환경 문제가 겹친 것이었다.

1. VS Code 디버그 설정이 `main.py`가 아니라 현재 열린 파일(`${file}`)을 실행하고 있었다.
2. VS Code가 바라보던 `.venv`에는 `torch` 등 필수 패키지가 설치되어 있지 않았다.
3. `app/predict.py`를 파일로 직접 실행하면 프로젝트 루트가 `PYTHONPATH`에 없어서 `from app...` import가 실패했다.
4. `app/predict.py` 안에서 `padded = pad_sequences(...)` 코드가 주석 뒤에 붙어 있어, 예측 함수 호출 시 다음 에러가 날 상태였다.
5. `.venv`와 `.venv-1` 두 가상환경이 섞여 있어서 어느 Python을 쓰는지 불명확했다.

현재는 `.venv` 하나로 통일했고, VS Code도 `.venv/Scripts/python.exe`를 사용하도록 정리했다.

## 발견된 문제

### 1. F5가 현재 파일을 실행하고 있었음

기존 `.vscode/launch.json` 설정은 다음 형태였다.

```json
"program": "${file}"
```

이 설정에서는 현재 에디터에 열려 있는 파일이 그대로 실행된다.

따라서 `app/predict.py`를 열고 F5를 누르면 프로젝트 진입점인 `main.py`가 아니라 `predict.py`만 실행된다. 원래 `predict.py`는 함수 정의 중심 파일이라, 단독 실행 시 기대한 학습/예측 흐름이 실행되지 않았다.

조치:

- `main.py`를 실행하는 디버그 구성을 추가했다.
- 현재 파일 실행 구성은 유지하되 `cwd`와 `PYTHONPATH`를 지정했다.

현재 설정:

```json
"program": "${workspaceFolder}/main.py",
"cwd": "${workspaceFolder}",
"env": {
    "PYTHONPATH": "${workspaceFolder}",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1"
}
```

### 2. 잘못된 가상환경을 사용하고 있었음

당시 상태:

- `.venv`: Anaconda Python 3.12 기반, 거의 빈 환경
- `.venv-1`: Python 3.14 기반, `torch`, `konlpy`, `scikit-learn` 등 실제 패키지가 설치된 환경

`.venv`로 실행하면 다음 에러가 발생했다.

```text
ModuleNotFoundError: No module named 'torch'
```

조치:

- 빈 `.venv` 제거
- `.venv-1`의 패키지 환경을 `.venv`로 통합
- `pip` 실행 파일과 activation 스크립트에 남아 있던 `.venv-1` 경로 정리
- `.vscode/settings.json`을 `.venv` 기준으로 수정

현재 확인 결과:

```text
.venv    존재함
.venv-1  존재하지 않음
Python   3.14.4
pip      .venv\Lib\site-packages 기준
torch    2.12.1+cpu
```

### 3. `app/predict.py` 직접 실행 시 import 경로가 깨졌음

`predict.py`는 내부에서 다음 import를 사용한다.

```python
from app.config import Config
from app.model import TextLSTMClassifier
from app.preprocess import clean_text, pad_sequences, texts_to_sequences
```

프로젝트 루트가 Python import 경로에 없으면 `python app/predict.py` 실행 시 다음 에러가 난다.

```text
ModuleNotFoundError: No module named 'app'
```

조치:

- VS Code 디버그 설정에 `"cwd": "${workspaceFolder}"` 추가
- VS Code 디버그 설정에 `"PYTHONPATH": "${workspaceFolder}"` 추가

### 4. `predict_text()` 내부에 실제 코드 버그가 있었음

기존 `app/predict.py`에서 `padded = pad_sequences(...)` 코드가 주석 뒤에 붙어 있었다.

문제 형태:

```python
sequence = texts_to_sequences([cleaned], metadata["vocab"])  # ... padded = pad_sequences(...)
```

이 경우 `padded` 변수는 실제로 생성되지 않는다. 따라서 예측 함수가 실행되면 `torch.tensor(padded, ...)` 부분에서 실패할 수 있었다.

조치:

```python
cleaned = clean_text(text)
sequence = texts_to_sequences([cleaned], metadata["vocab"])
padded = pad_sequences(sequence, config.max_len)
```

### 5. 한글 출력이 깨져 보였음

`main.py` 파일 자체는 UTF-8로 정상 저장되어 있었다. 다만 PowerShell/터미널 출력 인코딩 때문에 한글이 깨져 보였다.

조치:

VS Code 디버그 환경에 다음 설정을 추가했다.

```json
"PYTHONIOENCODING": "utf-8",
"PYTHONUTF8": "1"
```

## 현재 검증 결과

필수 패키지 import 확인:

```text
C:\Users\playdata2\Documents\llm_workspace\NAVER_NEWS_LSTM_Classifier\.venv\Scripts\python.exe
torch 2.12.1+cpu
imports ok
```

`app/predict.py` 직접 실행 확인:

```text
News: 삼성전자 반도체 기술 개발 성공 발표
Predicted category: IT  (probability: 0.4675)
```

모듈 실행 확인:

```powershell
python -m app.predict
```

결과:

```text
News: 삼성전자 반도체 기술 개발 성공 발표
Predicted category: IT  (probability: 0.4675)
```

## 현재 VS Code 실행 방법

전체 프로젝트 흐름 실행:

```text
Python: Run NAVER LSTM Classifier
```

주의: 이 구성은 `main.py`를 실행하므로 `train_model(config)`가 먼저 수행된다. 학습이 다시 돌기 때문에 오래 걸릴 수 있다.

빠른 예측 테스트:

```text
Python Debugger: Current File
```

이 구성은 현재 열린 파일을 실행한다. `app/predict.py`를 열고 실행하면 저장된 모델을 불러와 샘플 예측만 수행한다.

## 남은 주의점

`models/*.pt`, `models/*.pkl`, `models/*.png`는 `.gitignore`에 의해 Git 추적에서 제외되어 있다. 현재 로컬에는 모델 파일이 있어서 예측이 되지만, 새로 clone한 환경에서는 모델 파일이 없으면 `app/predict.py` 실행이 실패할 수 있다.

새 환경에서 실행하려면 둘 중 하나가 필요하다.

1. `main.py` 또는 학습 스크립트로 모델을 먼저 학습해서 `models/naver_lstm_model.pt`와 `models/naver_lstm_model_meta.pkl`을 생성한다.
2. 모델 아티팩트를 별도 저장소, 릴리스, 드라이브 등에서 내려받는 절차를 README에 명시한다.

또한 현재 Python은 3.14.4이다. 지금 실행은 정상이나, 수업/팀 환경 재현성을 높이려면 Python 버전을 README에 명시하는 것이 좋다.
