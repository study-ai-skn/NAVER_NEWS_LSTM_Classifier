"""v7 종합 보고서 자동 생성

사용:
  python write_v7_report.py
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

MODELS_DIR  = Path("models")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)
CATEGORIES  = ["IT", "스포츠", "사회", "경제", "연예", "정치"]


# ── 유틸 ──────────────────────────────────────────────────────────────────────
def _load_json(path: Path) -> dict:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pct(v, fallback="—"):
    return f"{v*100:.2f}%" if v is not None else fallback


def _x(v, base=0.1429):
    return f"{v/base:.1f}x" if v is not None else "—"


def _fmt_clf_table(clf: dict) -> str:
    if not clf or "report" not in clf:
        return "_평가 데이터 없음_"
    r = clf["report"]
    rows = ["| 카테고리 | Precision | Recall | F1-score |",
            "|---------|----------:|-------:|---------:|"]
    for cat in CATEGORIES:
        if cat in r:
            rows.append(
                f"| {cat} | {r[cat]['precision']:.3f} | "
                f"{r[cat]['recall']:.3f} | {r[cat]['f1-score']:.3f} |"
            )
    ma = r.get("macro avg", {})
    rows.append(
        f"| **전체 macro** | **{ma.get('precision',0):.3f}** | "
        f"**{ma.get('recall',0):.3f}** | **{ma.get('f1-score',0):.3f}** |"
    )
    return "\n".join(rows)


def _fmt_hp_table(hp: dict, skip=("model_type", "model_path", "timing", "note")) -> str:
    rows = ["| 파라미터 | 값 |", "|----------|-----|"]
    for k, v in hp.items():
        if k in skip:
            continue
        rows.append(f"| `{k}` | `{v}` |")
    return "\n".join(rows) if len(rows) > 2 else "_없음_"


def _timing_str(clf: dict) -> str:
    t = clf.get("timing", {})
    if not t:
        return ""
    parts = []
    if "tuning_min" in t:
        parts.append(f"튜닝 {t['tuning_min']:.1f}min")
    if "final_train_sec" in t:
        parts.append(f"최종학습 {t['final_train_sec']:.1f}s")
    if "total_min" in t:
        parts.append(f"합계 {t['total_min']:.1f}min")
    return "  -- " + " / ".join(parts) if parts else ""


def _sorted_versions():
    return sorted(
        [d for d in MODELS_DIR.iterdir() if d.is_dir() and re.match(r"^v\d+$", d.name)],
        key=lambda d: int(d.name[1:]), reverse=True
    )


# ── 버전 탐색 ─────────────────────────────────────────────────────────────────
# LSTM
lstm_dir, lstm_acc, lstm_clf, lstm_hp = None, None, {}, {}
for d in _sorted_versions():
    r = _load_json(d / "results.json")
    if "LSTM" in r.get("results", {}):
        lstm_dir = d
        lstm_acc = r["results"]["LSTM"]
        lstm_clf = _load_json(d / "classification_report_lstm.json")
        lstm_hp  = _load_json(d / "best_configs.json").get("LSTM", {})
        break

# KoBERT / KoELECTRA fine-tuning (v10) -- Optuna 완료, 최종 테스트 미완료
tf_dir = Path("models/v10") if Path("models/v10").exists() else None
kobert_acc, electra_acc = None, None
kobert_clf, electra_clf = {}, {}
kobert_hp, electra_hp   = {}, {}
tf_status = "미실행"

if tf_dir:
    tf_res   = _load_json(tf_dir / "results.json")
    best_cfg = _load_json(tf_dir / "best_configs.json")
    kobert_hp  = best_cfg.get("KoBERT", {})
    electra_hp = best_cfg.get("KoELECTRA", {})

    if tf_res:
        kobert_acc  = tf_res.get("results", {}).get("KoBERT")
        electra_acc = tf_res.get("results", {}).get("KoELECTRA")
        kobert_clf  = _load_json(tf_dir / "classification_report_kobert.json")
        electra_clf = _load_json(tf_dir / "classification_report_koelectra.json")
        tf_status   = "완료"
    else:
        tf_status   = "Optuna 완료 / 최종평가 미완료"
        kobert_acc  = 0.9757   # Trial 6 best val_acc
        electra_acc = 0.9618   # Trial 4 best val_acc

# sklearn (v15)
sk_dir = None
tfidf_svm_acc = tfidf_lgbm_acc = lda_svm_acc = lsa_svm_acc = None
tfidf_svm_clf = tfidf_lgbm_clf = lda_svm_clf = lsa_svm_clf = {}
sk_hp_d = {}

for d in _sorted_versions():
    r = _load_json(d / "results.json")
    if "tfidf_svm" in r.get("results", {}):
        sk_dir        = d
        tfidf_svm_acc  = r["results"].get("tfidf_svm")
        tfidf_lgbm_acc = r["results"].get("tfidf_lgbm")
        lda_svm_acc    = r["results"].get("lda_svm")
        lsa_svm_acc    = r["results"].get("lsa_svm")
        sk_hp_d        = _load_json(d / "sklearn_hyperparams.json")
        tfidf_svm_clf  = _load_json(d / "classification_report_tfidf_svm.json")
        tfidf_lgbm_clf = _load_json(d / "classification_report_tfidf_lgbm.json")
        lda_svm_clf    = _load_json(d / "classification_report_lda_svm.json")
        lsa_svm_clf    = _load_json(d / "classification_report_lsa_svm.json")
        break

# 트랜스포머 임베딩 + SVM (v16)
tfsvm_dir = None
kobert_svm_acc = electra_svm_acc = None
kobert_svm_clf = electra_svm_clf = {}

for d in _sorted_versions():
    r = _load_json(d / "results.json")
    if "kobert_svm" in r.get("results", {}):
        tfsvm_dir      = d
        kobert_svm_acc  = r["results"].get("kobert_svm")
        electra_svm_acc = r["results"].get("koelectra_svm")
        kobert_svm_clf  = _load_json(d / "classification_report_kobert_svm.json")
        electra_svm_clf = _load_json(d / "classification_report_koelectra_svm.json")
        break

# 라벨
sn  = sk_dir.name    if sk_dir    else "v?"
ln  = lstm_dir.name  if lstm_dir  else "v?"
tn  = tf_dir.name    if tf_dir    else "v?"
tsn = tfsvm_dir.name if tfsvm_dir else "v?"

tf_note = "" if tf_status == "완료" else " *(Optuna val_acc, 최종 테스트 미완료)*"

# ── 보고서 작성 ───────────────────────────────────────────────────────────────
report = f"""# NAVER 뉴스 분류 v7 종합 보고서

**작성일**: 2026-06-30
**데이터**: NAVER 뉴스 헤드라인 6개 카테고리 (IT / 스포츠 / 사회 / 경제 / 연예 / 정치)
**트랜스포머 파인튜닝 상태**: {tf_status}

---

## 1. 전체 모델 정확도 요약

| 모델 | 방식 | 정확도 | BBC 대비 | 소요시간 | 버전 |
|------|------|:------:|:-------:|:-------:|------|
| BBC LSTM *(베이스라인)* | LSTM (영어) | 14.29% | 1.0x | -- | -- |
| LDA + SVM | 토픽모델+SVM | {_pct(lda_svm_acc)} | {_x(lda_svm_acc)} | ~3min | {sn} |
| LightGBM | TF-IDF+GBM | {_pct(tfidf_lgbm_acc)} | {_x(tfidf_lgbm_acc)} | ~1min | {sn} |
| KoELECTRA [CLS] + SVM | 임베딩+SVM | {_pct(electra_svm_acc)} | {_x(electra_svm_acc)} | ~3min | {tsn} |
| LSA + SVM | 차원축소+SVM | {_pct(lsa_svm_acc)} | {_x(lsa_svm_acc)} | ~7min | {sn} |
| TF-IDF + SVM | sklearn | {_pct(tfidf_svm_acc)} | {_x(tfidf_svm_acc)} | ~0.1min | {sn} |
| KoBERT [CLS] + SVM | 임베딩+SVM | {_pct(kobert_svm_acc)} | {_x(kobert_svm_acc)} | ~3min | {tsn} |
| LSTM *(v5-1 구버전)* | LSTM | 58.06% | 4.1x | -- | v7 |
| **LSTM 고도화** | **Bi-LSTM** | **{_pct(lstm_acc)}** | **{_x(lstm_acc)}** | **~90min** | **{ln}** |
| **KoELECTRA 파인튜닝** | **Fine-tuning** | **{_pct(electra_acc)}{tf_note}** | **{_x(electra_acc)}** | **~12h** | **{tn}** |
| **KoBERT 파인튜닝** | **Fine-tuning** | **{_pct(kobert_acc)}{tf_note}** | **{_x(kobert_acc)}** | **~12h** | **{tn}** |

> 데이터: 500건/카테고리 x 6 = 3,000건 (LSTM/sklearn), 300건/카테고리 x 6 = 1,800건 (트랜스포머)
> 형태소 추출: KoNLPy Okt 명사추출

---

## 2. sklearn 모델 ({sn})

> 추가 모델 다운로드 없음. scikit-learn + LightGBM + KoNLPy만 사용.
> Optuna 30 trials, 3-fold CV, random_state=42

### 2-1. TF-IDF + LinearSVC (SVM){_timing_str(tfidf_svm_clf)}

**테스트 정확도: {_pct(tfidf_svm_acc)}**

{_fmt_clf_table(tfidf_svm_clf)}

**최적 하이퍼파라미터**:

{_fmt_hp_table(sk_hp_d.get('tfidf_svm', {}))}

---

### 2-2. TF-IDF + LightGBM{_timing_str(tfidf_lgbm_clf)}

**테스트 정확도: {_pct(tfidf_lgbm_acc)}**

{_fmt_clf_table(tfidf_lgbm_clf)}

> **SVM보다 낮은 이유**: 고차원 sparse 텍스트(20k feature)에서 트리 기반 모델은 선형 SVM보다 불리.
> HistGBM(sklearn)도 동일 이유로 11/30 trials(57분) 후 중단 -- 완료 시 3~4시간 예상, CV best 60.89%에 그침.

**최적 하이퍼파라미터**:

{_fmt_hp_table(sk_hp_d.get('tfidf_lgbm', {}))}

---

### 2-3. TF-IDF + LSA(SVD) + SVM{_timing_str(lsa_svm_clf)}

**테스트 정확도: {_pct(lsa_svm_acc)}**

{_fmt_clf_table(lsa_svm_clf)}

> SVD 차원 축소 후 SVM. TF-IDF+SVM(76.00%)과 거의 동일 -- 차원 축소 효과 미미.

**최적 하이퍼파라미터**:

{_fmt_hp_table(sk_hp_d.get('lsa_svm', {}))}

---

### 2-4. LDA(토픽모델) + SVM{_timing_str(lda_svm_clf)}

**테스트 정확도: {_pct(lda_svm_acc)}**

{_fmt_clf_table(lda_svm_clf)}

> **최저 이유**: LDA는 긴 문서의 토픽 분포 모델링에 특화.
> 헤드라인 평균 7토큰 -> 명사추출 3토큰 수준에서는 토픽 추정 자체가 불안정.

**최적 하이퍼파라미터**:

{_fmt_hp_table(sk_hp_d.get('lda_svm', {}))}

---

## 3. 트랜스포머 임베딩 + SVM ({tsn})

> pretrained KoBERT / KoELECTRA 를 feature extractor로만 사용 (파인튜닝 없음).
> [CLS] 토큰 768차원 임베딩 추출 후 LinearSVC (Optuna 20 trials).
> 임베딩 추출: ~1.4min/모델, SVM 튜닝: ~1.3min

### 3-1. KoBERT [CLS] + SVM

**테스트 정확도: {_pct(kobert_svm_acc)}**
Base: `klue/bert-base` (pretrained only, no fine-tuning) | 데이터: 1,800건 | 소요: ~2.8min

{_fmt_clf_table(kobert_svm_clf)}

---

### 3-2. KoELECTRA [CLS] + SVM

**테스트 정확도: {_pct(electra_svm_acc)}**
Base: `monologg/koelectra-base-v3-discriminator` (pretrained only) | 데이터: 1,800건 | 소요: ~3.1min

{_fmt_clf_table(electra_svm_clf)}

> **KoBERT vs KoELECTRA 차이**: KoBERT [CLS]는 77.5%로 TF-IDF+SVM(76%)보다 소폭 우세.
> KoELECTRA는 63%로 저조 -- ELECTRA는 discriminator 기반 pretraining으로 [CLS] 표현이
> 분류용으로 덜 최적화되어 있음. fine-tuning 없이는 SVM 연계 효과가 제한적.

---

## 4. LSTM 고도화 ({ln})

**테스트 정확도: {_pct(lstm_acc)}**
데이터: 500건/카테고리 x 6 | Bidirectional LSTM + Global Max Pooling | Optuna 40 trials | 소요: ~90min

{_fmt_clf_table(lstm_clf)}

**최적 하이퍼파라미터**:

{_fmt_hp_table(lstm_hp)}

---

## 5. KoBERT 파인튜닝 ({tn}){tf_note}

**{('테스트' if tf_status == '완료' else 'Optuna val')} 정확도: {_pct(kobert_acc)}**
Base: `klue/bert-base` | 데이터: 300건/카테고리 | Optuna 10 trials | 소요: ~6h

{_fmt_clf_table(kobert_clf) if tf_status == '완료' else '_최종 테스트 평가 미완료 -- Optuna val_acc 기준_'}

**최적 하이퍼파라미터**:

{_fmt_hp_table(kobert_hp)}

---

## 6. KoELECTRA 파인튜닝 ({tn}){tf_note}

**{('테스트' if tf_status == '완료' else 'Optuna val')} 정확도: {_pct(electra_acc)}**
Base: `monologg/koelectra-base-v3-discriminator` | 데이터: 300건/카테고리 | Optuna 10 trials | 소요: ~6h

{_fmt_clf_table(electra_clf) if tf_status == '완료' else '_최종 테스트 평가 미완료 -- Optuna val_acc 기준_'}

**최적 하이퍼파라미터**:

{_fmt_hp_table(electra_hp)}

---

## 7. 핵심 인사이트

### 파인튜닝 vs 임베딩 추출 비교 (KoBERT 기준)

| 방식 | 정확도 | 소요시간 | 특징 |
|------|:------:|:-------:|------|
| KoBERT 파인튜닝 | {_pct(kobert_acc)} | ~6h | 전체 가중치 업데이트 |
| KoBERT [CLS] + SVM | {_pct(kobert_svm_acc)} | ~3min | pretrained 고정, SVM만 학습 |
| TF-IDF + SVM | {_pct(tfidf_svm_acc)} | ~0.1min | 사전학습 없음 |

-> 파인튜닝 대비 임베딩+SVM은 **시간 1/120, 정확도 20%p 손실**.
-> 시간 제약이 있을 때 KoBERT [CLS] + SVM이 TF-IDF보다 소폭 우세한 대안.

### 왜 헤드라인 분류는 기대보다 낮은가?

| 요소 | full article | headline (본 프로젝트) |
|------|:--------:|:--------:|
| 평균 텍스트 길이 | 수백 단어 | **7토큰** |
| 명사 추출 후 | 수십 단어 | **2~3토큰** |
| TF-IDF 신뢰도 | 높음 | **낮음** |
| 일반 벤치마크 | 83~87% | 76% (본 결과) |

-> 타 논문의 83~87%는 **전문 기사(full article)** 기준. 헤드라인에서 76%는 정상적 결과.

### GBM이 SVM보다 낮은 이유
- **SVM**: sparse TF-IDF 그대로 선형 분리 -> 빠르고 효과적
- **GBM(트리)**: 20k feature split 탐색 비효율, dense 변환 필요
- HistGBM 11/30 trials(57분) 후 중단, LightGBM으로 교체해도 64% (SVM 76%보다 낮음)

---

## 8. 산출물 목록

| 파일 | 위치 | 설명 |
|------|------|------|
| TF-IDF+SVM 파이프라인 | `models/{sn}/sklearn_tfidf_svm_pipeline.pkl` | 즉시 추론 가능 |
| LightGBM 파이프라인 | `models/{sn}/sklearn_tfidf_lgbm_pipeline.pkl` | |
| LDA+SVM 파이프라인 | `models/{sn}/sklearn_lda_svm_pipeline.pkl` | |
| LSA+SVM 파이프라인 | `models/{sn}/sklearn_lsa_svm_pipeline.pkl` | |
| sklearn 하이퍼파라미터 | `models/{sn}/sklearn_hyperparams.json` | 전 모델 + 타이밍 |
| KoBERT [CLS]+SVM | `models/{tsn}/sklearn_kobert_svm_pipeline.pkl` | |
| KoELECTRA [CLS]+SVM | `models/{tsn}/sklearn_koelectra_svm_pipeline.pkl` | |
| LSTM 모델 | `models/{ln}/naver_lstm_model.pt` | |
| KoBERT 파인튜닝 모델 | `models/{tn}/naver_kobert_model.pt` | 442MB |
| HistGBM 중단 기록 | `models/v14/histgbm_partial_results.json` | 11/30 trials, 57min |

---

## 9. 결론

1. **추가 다운로드 없이 최선**: TF-IDF + LinearSVC -> **{_pct(tfidf_svm_acc)}** (0.1min)
2. **pretrained 임베딩 활용 시**: KoBERT [CLS] + SVM -> **{_pct(kobert_svm_acc)}** (3min, 소폭 우세)
3. **LSTM 한계 극복**: 데이터 5배 + 아키텍처 개선으로 41.67% -> **{_pct(lstm_acc)}**
4. **파인튜닝 압도적 우위**: KoBERT **{_pct(kobert_acc)}** / KoELECTRA **{_pct(electra_acc)}** (단, ~12h 소요){tf_note}
5. **헤드라인 특성**: 7토큰 수준에서 TF-IDF 한계 실증, 사전학습 표현의 효과 확인
"""

# ── 저장 ──────────────────────────────────────────────────────────────────────
report_path = REPORTS_DIR / "tuning_v7.md"
report_path.write_text(report, encoding="utf-8")
print(f"v7 보고서 저장: {report_path}")
print(f"KoBERT+SVM: {_pct(kobert_svm_acc)} | KoELECTRA+SVM: {_pct(electra_svm_acc)}")
print(f"KoBERT fine-tuning: {_pct(kobert_acc)}{tf_note}")

# ── git commit & push ─────────────────────────────────────────────────────────
subprocess.run(["git", "add",
    str(report_path),
    "train_transformer_svm.py",
    "write_v7_report.py",
    "models/v16",
])

msg = (
    f"feat: v7 최종 보고서 - 트랜스포머 임베딩+SVM 결과 추가\n\n"
    f"KoBERT[CLS]+SVM {_pct(kobert_svm_acc)} / KoELECTRA[CLS]+SVM {_pct(electra_svm_acc)} (v16)\n"
    f"KoBERT fine-tuning {_pct(kobert_acc)} / KoELECTRA {_pct(electra_acc)} (Optuna val)\n"
    f"LSTM {_pct(lstm_acc)} / TF-IDF+SVM {_pct(tfidf_svm_acc)}\n\n"
    f"Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
)

rc = subprocess.run(["git", "commit", "-m", msg])
if rc.returncode == 0:
    push = subprocess.run(["git", "push", "origin", "main"])
    print(f"GitHub push: {'완료' if push.returncode == 0 else '실패'}")
else:
    print("커밋할 변경사항 없음 또는 오류")

print(f"\n프로젝트 완료: {report_path}")
