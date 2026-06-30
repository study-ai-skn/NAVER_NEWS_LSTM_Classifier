"""v7 종합 보고서 자동 생성

v10 (KoBERT/KoELECTRA) + sklearn 모델 결과를 읽어
reports/tuning_v7.md 를 작성하고 git commit & push 합니다.

사용:
  python write_v7_report.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

MODELS_DIR  = Path("models")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _find_version(model_types: list[str]) -> Path | None:
    """results.json 에 지정된 model_type 이 모두 있는 최신 버전 폴더를 반환."""
    dirs = sorted(
        [d for d in MODELS_DIR.iterdir() if d.is_dir() and re.match(r"^v\d+$", d.name)],
        key=lambda d: int(d.name[1:]), reverse=True
    )
    for d in dirs:
        res = _load_json(d / "results.json")
        results = res.get("results", {})
        if all(mt in results for mt in model_types):
            return d
    return None


def _fmt_table(clf: dict, categories: list[str]) -> str:
    if not clf or "report" not in clf:
        return "_데이터 없음_"
    r = clf["report"]
    rows = ["| 카테고리 | Precision | Recall | F1 |",
            "|---------|----------:|-------:|---:|"]
    for cat in categories:
        if cat in r:
            rows.append(f"| {cat} | {r[cat]['precision']:.3f} | {r[cat]['recall']:.3f} | {r[cat]['f1-score']:.3f} |")
    ma = r.get("macro avg", {})
    rows.append(f"| **전체(macro)** | **{ma.get('precision',0):.3f}** | **{ma.get('recall',0):.3f}** | **{ma.get('f1-score',0):.3f}** |")
    return "\n".join(rows)


def _pct(v):
    return f"{v*100:.2f}%" if v is not None else "—"


# ── 버전 폴더 탐색 ─────────────────────────────────────────────────────────────
CATEGORIES = ["IT", "스포츠", "사회", "경제", "연예", "정치"]

# v9: LSTM 고도화
lstm_dir = _find_version(["LSTM"])
lstm_res  = _load_json(lstm_dir / "results.json") if lstm_dir else {}
lstm_acc  = lstm_res.get("results", {}).get("LSTM")
lstm_clf  = _load_json(lstm_dir / "classification_report_lstm.json") if lstm_dir else {}
lstm_hp   = _load_json(lstm_dir / "best_configs.json").get("LSTM", {}) if lstm_dir else {}

# v10: KoBERT + KoELECTRA
tf_dir      = _find_version(["KoBERT", "KoELECTRA"])
tf_res      = _load_json(tf_dir / "results.json") if tf_dir else {}
kobert_acc  = tf_res.get("results", {}).get("KoBERT")
electra_acc = tf_res.get("results", {}).get("KoELECTRA")
kobert_clf  = _load_json(tf_dir / "classification_report_kobert.json") if tf_dir else {}
electra_clf = _load_json(tf_dir / "classification_report_koelectra.json") if tf_dir else {}
kobert_hp   = _load_json(tf_dir / "best_configs.json").get("KoBERT", {}) if tf_dir else {}
electra_hp  = _load_json(tf_dir / "best_configs.json").get("KoELECTRA", {}) if tf_dir else {}

# sklearn 버전
sk_dir = None
for d in sorted(MODELS_DIR.iterdir(), key=lambda x: int(x.name[1:]), reverse=True):
    if not re.match(r"^v\d+$", d.name): continue
    res = _load_json(d / "results.json")
    if "tfidf_svm" in res.get("results", {}):
        sk_dir = d
        break

sk_res    = _load_json(sk_dir / "results.json") if sk_dir else {}
sk_hp     = _load_json(sk_dir / "sklearn_hyperparams.json") if sk_dir else {}
sk_results = sk_res.get("results", {})

tfidf_svm_acc  = sk_results.get("tfidf_svm")
tfidf_gbm_acc  = sk_results.get("tfidf_gbm")
lda_svm_acc    = sk_results.get("lda_svm")
lsa_svm_acc    = sk_results.get("lsa_svm")

tfidf_svm_clf = _load_json(sk_dir / "classification_report_tfidf_svm.json") if sk_dir else {}
tfidf_gbm_clf = _load_json(sk_dir / "classification_report_tfidf_gbm.json") if sk_dir else {}
lda_svm_clf   = _load_json(sk_dir / "classification_report_lda_svm.json") if sk_dir else {}
lsa_svm_clf   = _load_json(sk_dir / "classification_report_lsa_svm.json") if sk_dir else {}


# ── 하이퍼파라미터 테이블 포맷 ────────────────────────────────────────────────
def _hp_table(hp: dict, skip_keys: list[str] = ["model_type", "model_path"]) -> str:
    rows = ["| 파라미터 | 값 |", "|----------|-----|"]
    for k, v in hp.items():
        if k in skip_keys:
            continue
        rows.append(f"| `{k}` | `{v}` |")
    return "\n".join(rows) if len(rows) > 2 else "_없음_"


# ── 리포트 본문 ────────────────────────────────────────────────────────────────
now_version_info = (
    f"LSTM: {lstm_dir.name if lstm_dir else '?'}  /  "
    f"Transformer: {tf_dir.name if tf_dir else '?'}  /  "
    f"sklearn: {sk_dir.name if sk_dir else '?'}"
)

report = f"""# v7 종합 보고서 — 전 모델 비교 분석

**작성일**: 2026-06-30
**버전**: {now_version_info}
**데이터**: NAVER 뉴스 500건/카테고리 × 6 = 3,000건 (LSTM·sklearn)
**분류 카테고리**: IT / 스포츠 / 사회 / 경제 / 연예 / 정치

---

## 1. 전체 모델 정확도 비교

| 모델 | 방식 | 테스트 정확도 | BBC 대비 | 버전 |
|------|------|:------------:|:--------:|------|
| BBC LSTM (베이스라인) | LSTM (영어) | 14.29% | 1.0× | — |
| 형태소 TF-IDF + LDA + SVM | 예외케이스 | {_pct(lda_svm_acc)} | {f'{lda_svm_acc/0.1429:.1f}×' if lda_svm_acc else '—'} | {sk_dir.name if sk_dir else '?'} |
| 형태소 TF-IDF + HistGBM | sklearn | {_pct(tfidf_gbm_acc)} | {f'{tfidf_gbm_acc/0.1429:.1f}×' if tfidf_gbm_acc else '—'} | {sk_dir.name if sk_dir else '?'} |
| 형태소 TF-IDF + LSA + SVM | 예외케이스 | {_pct(lsa_svm_acc)} | {f'{lsa_svm_acc/0.1429:.1f}×' if lsa_svm_acc else '—'} | {sk_dir.name if sk_dir else '?'} |
| 형태소 TF-IDF + SVM | sklearn | {_pct(tfidf_svm_acc)} | {f'{tfidf_svm_acc/0.1429:.1f}×' if tfidf_svm_acc else '—'} | {sk_dir.name if sk_dir else '?'} |
| NAVER LSTM v5-1 (이전 최고) | LSTM | 58.06% | 4.1× | v7폴더 |
| **NAVER LSTM 고도화** | **LSTM** | **{_pct(lstm_acc)}** | **{f'{lstm_acc/0.1429:.1f}×' if lstm_acc else '—'}** | **{lstm_dir.name if lstm_dir else '?'}** |
| **KoBERT** | **Fine-tuning** | **{_pct(kobert_acc)}** | **{f'{kobert_acc/0.1429:.1f}×' if kobert_acc else '—'}** | **{tf_dir.name if tf_dir else '?'}** |
| **KoELECTRA** | **Fine-tuning** | **{_pct(electra_acc)}** | **{f'{electra_acc/0.1429:.1f}×' if electra_acc else '—'}** | **{tf_dir.name if tf_dir else '?'}** |

---

## 2. sklearn 모델 결과 ({sk_dir.name if sk_dir else '?'})

> 추가 다운로드 없이 설치된 패키지(scikit-learn, KoNLPy)만 사용
> 데이터: 500건/카테고리, Optuna {sk_hp.get('tfidf_svm', {}).get('cv_best_acc', '?')} val_acc 기준
> 재현: `random_state=42`, `python train_sklearn_models.py`

### 2-1. 형태소 TF-IDF + LinearSVC (SVM) — 메인

{_fmt_table(tfidf_svm_clf, CATEGORIES)}

**최적 하이퍼파라미터**:
{_hp_table(sk_hp.get('tfidf_svm', {}))}

### 2-2. 형태소 TF-IDF + HistGradientBoosting (GBM) — 메인

{_fmt_table(tfidf_gbm_clf, CATEGORIES)}

**최적 하이퍼파라미터**:
{_hp_table(sk_hp.get('tfidf_gbm', {}))}

### 2-3. 형태소 CountVec + LDA + SVM — 예외 케이스

> LDA는 긴 문서에 최적화된 토픽 모델. 헤드라인(평균 7토큰)에서는 불리.

{_fmt_table(lda_svm_clf, CATEGORIES)}

**최적 하이퍼파라미터**:
{_hp_table(sk_hp.get('lda_svm', {}))}

### 2-4. 형태소 TF-IDF + LSA(SVD) + SVM — 예외 케이스

> TF-IDF 희소 벡터를 SVD로 차원 축소 후 SVM 적용.

{_fmt_table(lsa_svm_clf, CATEGORIES)}

**최적 하이퍼파라미터**:
{_hp_table(sk_hp.get('lsa_svm', {}))}

---

## 3. LSTM 고도화 결과 ({lstm_dir.name if lstm_dir else '?'})

**테스트 정확도: {_pct(lstm_acc)}**
데이터: 500건/카테고리 × 6 = 3,000건 | Optuna 40 trials | 재현: `python naver_lstm_classifier.py --models LSTM --n_trials 40 --max-items 500`

{_fmt_table(lstm_clf, CATEGORIES)}

**최적 하이퍼파라미터**:
{_hp_table(lstm_hp)}

---

## 4. KoBERT 결과 ({tf_dir.name if tf_dir else '?'})

**테스트 정확도: {_pct(kobert_acc)}**
Base: `klue/bert-base` | 데이터: 300건/카테고리 | Optuna 10 trials
재현: `python naver_lstm_classifier.py --models KoBERT --n_trials 10`

{_fmt_table(kobert_clf, CATEGORIES)}

**최적 하이퍼파라미터**:
{_hp_table(kobert_hp)}

---

## 5. KoELECTRA 결과 ({tf_dir.name if tf_dir else '?'})

**테스트 정확도: {_pct(electra_acc)}**
Base: `monologg/koelectra-base-v3-discriminator` | 데이터: 300건/카테고리 | Optuna 10 trials
재현: `python naver_lstm_classifier.py --models KoELECTRA --n_trials 10`

{_fmt_table(electra_clf, CATEGORIES)}

**최적 하이퍼파라미터**:
{_hp_table(electra_hp)}

---

## 6. 모델 방법론 비교 분석

### 왜 같은 데이터인데 정확도가 다른가?

| 요소 | sklearn (TF-IDF) | LSTM | KoBERT/KoELECTRA |
|------|:----------------:|:----:|:----------------:|
| 단어 의미 이해 | ❌ 빈도만 | △ 학습 중 습득 | ✅ 사전학습 |
| 단어 순서 활용 | ❌ | ✅ | ✅ |
| 서브워드 처리 | ❌ | ❌ | ✅ (WordPiece) |
| 문장 전체 문맥 | ❌ | △ (단방향) | ✅ (Attention) |
| 데이터 의존성 | 중간 | 높음 | 낮음 |
| 학습 시간 | **~10분** | ~90분 | ~10시간 |
| 추가 다운로드 | ❌ | ❌ | ✅ (모델) |

### sklearn 예외 케이스 결론

| 모델 | 예측 | 실제 | 이유 |
|------|:----:|:----:|------|
| LDA + SVM | 낮음 | {_pct(lda_svm_acc)} | 짧은 헤드라인에 LDA 부적합 (긴 문서 전용) |
| LSA + SVM | 중간 | {_pct(lsa_svm_acc)} | 차원 축소로 일부 정보 손실 |

---

## 7. 산출물 재현 가이드

| 모델 | 실행 명령 | 산출물 위치 |
|------|-----------|-------------|
| sklearn 전체 | `python train_sklearn_models.py` | `models/{sk_dir.name if sk_dir else 'v?'}/sklearn_*` |
| LSTM | `python naver_lstm_classifier.py --models LSTM --n_trials 40 --max-items 500` | `models/{lstm_dir.name if lstm_dir else 'v?'}/` |
| KoBERT | `python naver_lstm_classifier.py --models KoBERT --n_trials 10` | `models/{tf_dir.name if tf_dir else 'v?'}/` |
| KoELECTRA | `python naver_lstm_classifier.py --models KoELECTRA --n_trials 10` | `models/{tf_dir.name if tf_dir else 'v?'}/` |

모든 파이프라인은 `pickle` 저장되어 로드 후 즉시 추론 가능:
```python
import pickle
with open("models/{sk_dir.name if sk_dir else 'v?'}/sklearn_tfidf_svm_pipeline.pkl", "rb") as f:
    obj = pickle.load(f)
pipeline, le = obj["pipeline"], obj["label_encoder"]
pred = le.inverse_transform(pipeline.predict(["삼성전자 반도체 신제품 발표"]))
```

---

## 8. 결론

1. **추가 다운로드 없이 최선**: 형태소 TF-IDF + LinearSVC → {_pct(tfidf_svm_acc)}
2. **LSTM 한계 극복**: 데이터 5배 증가(100→500건/cat)로 41.67% → {_pct(lstm_acc)} 달성
3. **트랜스포머 우위 확인**: KoBERT {_pct(kobert_acc)} / KoELECTRA {_pct(electra_acc)}
4. **LDA 예외 케이스 검증**: 단문 헤드라인에 LDA 토픽 모델링 비적합 실증

다음 단계: KoBERT + KoELECTRA Soft Voting 앙상블로 추가 향상 도전
"""

# 저장
report_path = REPORTS_DIR / "tuning_v7.md"
report_path.write_text(report, encoding="utf-8")
print(f"v7 보고서 저장: {report_path}")

# git commit & push
print("\nGit commit & push 중...")
subprocess.run(["git", "add",
    str(report_path),
    str(sk_dir) if sk_dir else ".",
    "train_sklearn_models.py",
    "write_v7_report.py",
])
commit_msg = (
    f"docs: v7 종합 보고서 - 전 모델 비교\n\n"
    f"LSTM {_pct(lstm_acc)} / KoBERT {_pct(kobert_acc)} / "
    f"KoELECTRA {_pct(electra_acc)}\n"
    f"sklearn: TF-IDF+SVM {_pct(tfidf_svm_acc)} / "
    f"TF-IDF+GBM {_pct(tfidf_gbm_acc)}\n\n"
    "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
)
subprocess.run(["git", "commit", "-m", commit_msg])
rc = subprocess.run(["git", "push", "origin", "main"])
print(f"GitHub push: {'완료' if rc.returncode == 0 else '실패'}")
print(f"\n완료: {report_path}")
