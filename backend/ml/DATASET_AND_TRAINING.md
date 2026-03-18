# WordCraft NLP Dataset and Reranker Training

This folder contains the first production-ready scaffold for building a supervised dataset across all WordCraft features and training a reranker model.

## What is included

- `backend/ml/data/dataset_schema.json`: canonical sample schema
- `backend/ml/data/seed_gold.jsonl`: seed gold samples across editor + tools
- `backend/ml/scripts/build_dataset.py`: expands seed samples with model candidates and creates labeled ranker data
- `backend/ml/scripts/split_dataset.py`: creates frozen group-wise train/val/test/regression splits (`random` or `hard`)
- `backend/ml/scripts/train_reranker.py`: trains a learning-to-rank classifier on candidate relevance labels
- `backend/ml/scripts/eval_reranker.py`: evaluates artifact on labeled JSONL with ranking metrics
- `backend/ml/scripts/test_reranker.py`: evaluates the artifact on frozen unseen test split
- `backend/ml/scripts/ab_eval.py`: compares baseline rank (`model_score`) vs reranker on same unseen split
- `backend/ml/scripts/export_feedback_dataset.py`: converts MongoDB user ratings into incremental dataset rows
- `backend/ml/hard_negative.py`: hard-negative mining (POS-compatible near-miss negatives)
- `backend/ml/retrieval.py`: semantic retrieval layer (optional FAISS, numpy fallback)
- `backend/ml/scripts/metrics_dashboard.py`: task-level metrics (nDCG@1/3/5, MRR, Hit@k, constraint-pass)
- `backend/ml/scripts/build_behavior_priors.py`: converts ratings + implicit actions to behavior priors JSON
- `backend/ml/scripts/train_cross_encoder.py`: neural reranker training (listwise/pairwise objective)
- `backend/ml/scripts/train_contrastive_ranker.py`: triplet contrastive training scaffold
- `backend/ml/scripts/train_rewrite_lora.py`: LoRA rewrite fine-tuning scaffold

## Recommended workflow (dataset-once + incremental updates)

1. Build your base dataset once:
   - `python -m backend.ml.scripts.build_dataset --inject-gold-positives false --hard-negatives true`
2. Create frozen split files:
   - `python -m backend.ml.scripts.split_dataset --split-mode hard`
3. Train on train split and evaluate on val split:
   - `python -m backend.ml.scripts.train_reranker --use-default-splits --exclude-gold-seed`
   - `python -m backend.ml.scripts.eval_reranker --use-default-val --exclude-gold-seed`
4. Test on unseen frozen test split:
   - `python -m backend.ml.scripts.test_reranker --exclude-gold-seed`
5. Run A/B comparison on unseen test:
   - `python -m backend.ml.scripts.ab_eval --exclude-gold-seed`
6. Per-task dashboard:
   - `python -m backend.ml.scripts.metrics_dashboard --dataset backend/ml/data/splits/test.jsonl --artifact backend/ml/models/reranker.pkl --exclude-gold-seed`
7. Collect user feedback in MongoDB (`/feedback` ratings + `/feedback/implicit` inserted/copied/favorited).
8. Build behavior priors (for runtime 0.7 ML + 0.3 behavior blend):
   - `python -m backend.ml.scripts.build_behavior_priors`
9. Only when needed, export/merge feedback rows:
   - `python -m backend.ml.scripts.export_feedback_dataset --append-default`
10. Rebuild/split/retrain only when developers decide to incorporate new feedback rows.

## Labeling policy

Candidate labels are interpreted as:

- `3`: best output for that sample (top quality)
- `2`: clearly valid/relevant output
- `1`: acceptable but weaker output
- `0`: invalid or irrelevant output

For ranking metrics (`Hit@k`, `MRR`, `nDCG@5`), labels `>= 2` are treated as relevant.

## Commands

From repository root:

```bash
python -m backend.ml.scripts.build_dataset
python -m backend.ml.scripts.split_dataset --split-mode hard
python -m backend.ml.scripts.train_reranker --use-default-splits --exclude-gold-seed
python -m backend.ml.scripts.eval_reranker --use-default-val --exclude-gold-seed
python -m backend.ml.scripts.test_reranker --exclude-gold-seed
python -m backend.ml.scripts.ab_eval --exclude-gold-seed
python -m backend.ml.scripts.export_feedback_dataset --append-default
python -m backend.ml.scripts.build_behavior_priors
python -m backend.ml.scripts.metrics_dashboard --dataset backend/ml/data/splits/test.jsonl --artifact backend/ml/models/reranker.pkl --exclude-gold-seed
```

Optional args:

```bash
python -m backend.ml.scripts.build_dataset --seed backend/ml/data/seed_gold.jsonl --output backend/ml/data/dataset_ranker.jsonl --limit 12
python -m backend.ml.scripts.split_dataset --dataset backend/ml/data/dataset_ranker.jsonl --out-dir backend/ml/data/splits --split-mode hard --seed 42
python -m backend.ml.scripts.train_reranker --dataset backend/ml/data/dataset_ranker.jsonl --artifact backend/ml/models/reranker.pkl --test-size 0.25 --exclude-gold-seed
python -m backend.ml.scripts.train_reranker --train backend/ml/data/splits/train.jsonl --val backend/ml/data/splits/val.jsonl --artifact backend/ml/models/reranker.pkl --exclude-gold-seed
python -m backend.ml.scripts.eval_reranker --dataset backend/ml/data/dataset_ranker.jsonl --artifact backend/ml/models/reranker.pkl --exclude-gold-seed
python -m backend.ml.scripts.test_reranker --dataset backend/ml/data/splits/test.jsonl --artifact backend/ml/models/reranker.pkl --exclude-gold-seed
python -m backend.ml.scripts.ab_eval --dataset backend/ml/data/splits/test.jsonl --artifact backend/ml/models/reranker.pkl --exclude-gold-seed
python -m backend.ml.scripts.export_feedback_dataset --output backend/ml/data/dataset_feedback.jsonl --append-to backend/ml/data/dataset_ranker.jsonl
python -m backend.ml.scripts.train_cross_encoder --train backend/ml/data/splits/train.jsonl --val backend/ml/data/splits/val.jsonl --objective listwise --exclude-gold-seed
python -m backend.ml.scripts.train_contrastive_ranker --train backend/ml/data/splits/train.jsonl --exclude-gold-seed
python -m backend.ml.scripts.train_rewrite_lora --dataset backend/ml/data/rewrite_train.jsonl --output-dir backend/ml/models/rewrite_lora
```

## Runtime knobs (new)

- `WORDCRAFT_DISABLE_RERANKER=1`: disables ML reranking.
- `WORDCRAFT_ENABLE_RETRIEVAL=1`: enables semantic retrieval candidate expansion before reranking (default off).
- `WORDCRAFT_RERANKER_ARTIFACT=backend/ml/models/reranker.pkl`: artifact path.
- `WORDCRAFT_BEHAVIOR_PRIORS=backend/ml/models/behavior_priors.json`: behavior priors path.
- `WORDCRAFT_CONFIDENCE_TEMPERATURE=1.0`: confidence calibration temperature.
- `WORDCRAFT_NORMALIZE_FINAL_SCORE=1`: min-max normalize final scores per request.

## Notes

- Cross-encoder artifacts are supported by eval/test/A-B scripts in addition to sklearn artifacts.
- LoRA training requires extra dependencies:
  - `pip install transformers datasets peft accelerate bitsandbytes`
- FAISS is optional:
  - `pip install faiss-cpu`
  - If unavailable, retrieval uses numpy fallback.

## Scaling to a large dataset

1. Add new JSONL rows in `seed_gold.jsonl` for each feature/mode.
2. Focus on hard cases:
   - blank POS disambiguation (ADV/ADJ/VERB)
   - context shifts (`nostalgia` vs `horror`)
   - smart match strict and fallback behavior
   - one-word substitution for person/quality/abstract phrases
3. Include explicit negatives for common failure patterns.
4. Keep at least 20-30% human-reviewed samples.

## Suggested acceptance targets

- `Hit@1 >= 0.65`
- `Hit@3 >= 0.80`
- `nDCG@5 >= 0.75`
- `MRR >= 0.70`
- Mode-specific regression tests remain green (`backend/tests/*`)
