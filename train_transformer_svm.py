"""트랜스포머 임베딩 + SVM

KoBERT / KoELECTRA 를 feature extractor 로 사용해
[CLS] 토큰 임베딩(768d)을 추출하고 LinearSVC 를 붙입니다.

산출물 (models/v{N}/):
  classification_report_kobert_svm.json
  classification_report_koelectra_svm.json
  sklearn_kobert_svm_pipeline.pkl  (SVM 부분만, 임베딩 재현 포함)
  results.json
"""

from __future__ import annotations

import json
import pickle
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from transformers import AutoTokenizer, AutoModel

from app.data import load_sample_data

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
MAX_ITEMS    = 300
N_TRIALS     = 20
CV_FOLDS     = 3
MODELS_DIR   = Path("models")
DEVICE       = "cpu"

TRANSFORMER_MODELS = {
    "kobert":    "klue/bert-base",
    "koelectra": "monologg/koelectra-base-v3-discriminator",
}

SCRIPT_START = time.time()


def _next_run_dir() -> Path:
    existing = sorted(
        int(d.name[1:]) for d in MODELS_DIR.iterdir()
        if d.is_dir() and re.match(r"^v\d+$", d.name)
    )
    run_dir = MODELS_DIR / f"v{(existing[-1] + 1) if existing else 1}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def extract_cls_embeddings(texts: list[str], model_name: str,
                            max_len: int = 64, batch_size: int = 32) -> np.ndarray:
    """[CLS] 토큰 임베딩 추출 (pretrained, no fine-tuning)."""
    from tqdm import tqdm
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()

    all_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size),
                      desc=f"  [{model_name.split('/')[-1]}] 임베딩 추출",
                      file=sys.stdout, ncols=72):
            batch = texts[i:i+batch_size]
            enc   = tokenizer(batch, truncation=True, padding=True,
                              max_length=max_len, return_tensors="pt")
            enc   = {k: v.to(DEVICE) for k, v in enc.items()}
            out   = model(**enc)
            cls   = out.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS]
            all_embs.append(cls)

    return np.vstack(all_embs)


def run_svm_on_embeddings(name: str, X_emb: np.ndarray,
                           y_tr, y_te, idx_tr, idx_te,
                           category_names: list[str], run_dir: Path,
                           embed_time: float) -> dict:
    """Optuna로 SVM C 튜닝 후 평가."""
    from tqdm import tqdm

    X_tr_emb = X_emb[idx_tr]
    X_te_emb = X_emb[idx_te]

    print(f"\n  [{name}] LinearSVC 튜닝 ({N_TRIALS} trials)...")
    t_tune = time.time()

    def objective(trial):
        C   = trial.suggest_float("C", 0.01, 100, log=True)
        clf = LinearSVC(C=C, max_iter=3000, random_state=RANDOM_STATE)
        return cross_val_score(clf, X_tr_emb, y_tr, cv=CV_FOLDS, scoring="accuracy").mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    pbar  = tqdm(total=N_TRIALS, desc=f"  [{name}_svm]", ncols=72, file=sys.stdout)

    def cb(s, t):
        pbar.update(1)
        pbar.set_postfix({"best": f"{s.best_value:.4f}",
                          "cur":  f"{t.value:.4f}" if t.value else "—"})
        sys.stdout.flush()

    study.optimize(objective, n_trials=N_TRIALS, callbacks=[cb])
    pbar.close()
    tune_sec = time.time() - t_tune

    best_C = study.best_params["C"]
    clf    = LinearSVC(C=best_C, max_iter=3000, random_state=RANDOM_STATE)

    t_train = time.time()
    clf.fit(X_tr_emb, y_tr)
    train_sec = time.time() - t_train

    preds = clf.predict(X_te_emb)
    acc   = accuracy_score(y_te, preds)
    clf_report = classification_report(
        y_te, preds, target_names=category_names, zero_division=0, output_dict=True
    )

    print(f"\n{'='*50}")
    print(f"  [{name}_svm] 테스트 정확도: {acc:.4f}  "
          f"(임베딩 {embed_time/60:.1f}min + 튜닝 {tune_sec/60:.1f}min + 학습 {train_sec:.1f}s)")
    print(f"{'='*50}")
    print(classification_report(y_te, preds, target_names=category_names, zero_division=0))

    total_sec = embed_time + tune_sec + train_sec
    timing = {
        "embedding_sec":   round(embed_time, 2),
        "tuning_sec":      round(tune_sec, 2),
        "final_train_sec": round(train_sec, 2),
        "total_sec":       round(total_sec, 2),
        "total_min":       round(total_sec / 60, 2),
    }

    rpt_path = run_dir / f"classification_report_{name}_svm.json"
    with open(rpt_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": f"{name}_svm",
            "base_model": TRANSFORMER_MODELS[name],
            "approach": "pretrained feature extraction + LinearSVC (no fine-tuning)",
            "accuracy": acc,
            "best_C": best_C,
            "report": clf_report,
            "timing": timing,
        }, f, ensure_ascii=False, indent=2)

    pkl_path = run_dir / f"sklearn_{name}_svm_pipeline.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "classifier": clf,
            "label_encoder_classes": category_names,
            "base_model": TRANSFORMER_MODELS[name],
            "best_C": best_C,
            "note": "임베딩은 extract_cls_embeddings() 로 재현"
        }, f)

    print(f"  저장 완료: {rpt_path.name}")
    return {"acc": acc, "timing": timing, "best_C": best_C}


# ── 메인 ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  트랜스포머 임베딩 + SVM")
print("  (pretrained KoBERT / KoELECTRA -> [CLS] -> LinearSVC)")
print("=" * 60)

texts, labels = load_sample_data(max_items=MAX_ITEMS)
le = LabelEncoder()
y  = le.fit_transform(labels)
category_names = list(le.classes_)

idx_all = np.arange(len(texts))
idx_trainval, idx_test = train_test_split(
    idx_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=0.2, random_state=RANDOM_STATE, stratify=y[idx_trainval])

y_tr = y[idx_train]
y_te = y[idx_test]

run_dir    = _next_run_dir()
all_results = {}
all_timing  = {}
print(f"\n산출물 저장 위치: {run_dir}")
print(f"데이터: {len(texts)}건 | Train {len(idx_train)} / Test {len(idx_test)}\n")

for model_key, model_path in TRANSFORMER_MODELS.items():
    print(f"\n{'='*60}")
    print(f"  [{model_key.upper()}] {model_path}")
    print(f"{'='*60}")

    t_emb = time.time()
    X_emb = extract_cls_embeddings(texts, model_path)
    embed_sec = time.time() - t_emb
    print(f"  임베딩 추출 완료: {X_emb.shape}  ({embed_sec/60:.1f}min)")

    res = run_svm_on_embeddings(
        model_key, X_emb,
        y_tr, y_te, idx_train, idx_test,
        category_names, run_dir, embed_sec
    )
    all_results[f"{model_key}_svm"] = res["acc"]
    all_timing[f"{model_key}_svm"]  = res["timing"]

# ── 결과 저장 ─────────────────────────────────────────────────────────────────
total_sec = time.time() - SCRIPT_START
results_path = run_dir / "results.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump({
        "version":   run_dir.name,
        "approach":  "pretrained transformer embedding + LinearSVC",
        "data":      f"data/naver_news_{MAX_ITEMS}per_cat.csv",
        "results":   all_results,
        "timing":    {**all_timing, "total_script_min": round(total_sec/60, 2)},
    }, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"  [{run_dir.name}] 트랜스포머 임베딩 + SVM 결과")
print(f"{'='*60}")
for name, acc in sorted(all_results.items(), key=lambda x: -x[1]):
    t   = all_timing[name]
    bar = "#" * int(acc * 20)
    print(f"  {name:20s}: {acc:.4f}  {bar}  ({t['total_min']:.1f}min)")
print(f"\n  전체 소요: {total_sec/60:.1f}min")
print(f"  산출물: {run_dir}")
print(f"{'='*60}\n")
