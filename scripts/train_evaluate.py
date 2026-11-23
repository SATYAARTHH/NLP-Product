#!/usr/bin/env python3
"""Train M3 baselines, evaluate M4 outcomes, and write reproducible artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from triageguard.modeling.classifiers import TemperatureScaledClassifier, text_classifier  # noqa: E402
from triageguard.modeling.fields import FieldExtractor  # noqa: E402
from triageguard.modeling.metrics import (  # noqa: E402
    classification_metrics,
    confidence_gate,
    field_metrics,
    pii_metrics,
    pii_string_metrics,
)
from triageguard.modeling.redaction import Redactor  # noqa: E402
from triageguard.modeling.routing import HybridRouting, RoutingRules  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_best(train: list[dict], val: list[dict], target: str):
    texts = [doc["text"] for doc in train]
    labels = [doc[target] for doc in train]
    val_texts = [doc["text"] for doc in val]
    val_labels = [doc[target] for doc in val]
    experiments = {}
    fitted = {}
    for name, use_chars in (("word_tfidf", False), ("word_char_tfidf", True)):
        model = text_classifier(character_features=use_chars)
        model.fit(texts, labels)
        prediction = model.predict(val_texts)
        score = f1_score(val_labels, prediction, average="macro")
        experiments[name] = round(float(score), 4)
        fitted[name] = model
    selected_name = max(experiments, key=experiments.get)
    selected = TemperatureScaledClassifier(fitted[selected_name]).fit_temperature(val_texts, val_labels)
    return selected, {"validation_macro_f1": experiments, "selected": selected_name, "temperature": round(selected.temperature, 3)}


def fit_route(train: list[dict], val: list[dict], fields: FieldExtractor, document_model):
    learned, experiments = fit_best(train, val, "route")
    val_texts = [doc["text"] for doc in val]
    val_labels = [doc["route"] for doc in val]
    rules = RoutingRules(fields, document_model).fit(train)
    rule_prediction = rules.predict(val_texts)
    rule_score = float(f1_score(val_labels, rule_prediction, average="macro"))
    experiments["validation_macro_f1"]["visible_fact_rules"] = round(rule_score, 4)
    hybrid = HybridRouting(rules, learned)
    hybrid_score = float(f1_score(val_labels, hybrid.predict(val_texts), average="macro"))
    experiments["validation_macro_f1"]["hybrid_rules_text"] = round(hybrid_score, 4)
    candidates = {"text": learned, "visible_fact_rules": rules, "hybrid_rules_text": hybrid}
    candidate_scores = {
        "text": max(experiments["validation_macro_f1"]["word_tfidf"], experiments["validation_macro_f1"]["word_char_tfidf"]),
        "visible_fact_rules": rule_score,
        "hybrid_rules_text": hybrid_score,
    }
    selected_name = max(candidate_scores, key=candidate_scores.get)
    if selected_name == "text":
        return learned, experiments
    selected = TemperatureScaledClassifier(candidates[selected_name]).fit_temperature(val_texts, val_labels)
    experiments["selected"] = selected_name
    experiments["temperature"] = round(selected.temperature, 3)
    return selected, experiments


def evaluate_classifier(model, documents: list[dict], target: str) -> tuple[dict, np.ndarray, np.ndarray]:
    texts = [doc["text"] for doc in documents]
    truth = np.asarray([doc[target] for doc in documents])
    probabilities = model.predict_proba(texts)
    prediction = model.classes_[probabilities.argmax(axis=1)]
    return classification_metrics(truth, prediction, probabilities, model.classes_), prediction, probabilities.max(axis=1)


def apply_gate(gate: dict, doc_truth, doc_pred, doc_conf, route_truth, route_pred, route_conf) -> dict:
    confidence = np.minimum(doc_conf, route_conf)
    accepted = confidence >= gate["threshold"]
    correct = (np.asarray(doc_truth) == doc_pred) & (np.asarray(route_truth) == route_pred)
    return {
        "coverage": round(float(accepted.mean()), 4),
        "joint_accuracy": round(float(correct[accepted].mean()), 4) if accepted.any() else None,
        "accepted": int(accepted.sum()),
        "total": len(accepted),
    }


def md_table(rows, headers):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(out)


def make_report(result: dict) -> str:
    rows = []
    for split in (
        "test_golden", "test_hard", "real_ocr",
        "control_test_golden", "control_test_hard", "control_real_ocr",
    ):
        score = result["evaluation"][split]
        rows.append([
            split,
            score["pii"].get("leak_recall", score["pii"].get("surviving_string_recall")),
            score["document_type"]["macro_f1"],
            score["fields"]["f1"],
            score["routing"]["per_class"]["high_value"]["recall"],
            score["routing"]["per_class"]["suspicious"]["recall"],
            score["routing"]["ece_15"],
            score["gate"]["coverage"],
        ])
    status = result["target_status"]
    golden_route = result["evaluation"]["test_golden"]["routing"]["per_class"]
    target_rows = [[name, values["target"], values["observed"], "PASS" if values["pass"] else "MISS"] for name, values in status.items()]
    return f"""# M3 model outcomes and M4 baseline evaluation

Generated by `scripts/train_evaluate.py` from frozen corpus version `{result['data']['corpus_version']}`.

## Selected models

- Document type: {result['experiments']['document_type']['selected']} + multinomial logistic regression.
- Routing: {result['experiments']['routing']['selected']} (visible-fact rules combined with a calibrated text classifier when selected).
- PII: high-recall context and format rules.
- Fields: anchor rules plus a training-only value lexicon. List items are scored as individual normalized values.
- Probabilities: temperature scaling chosen on validation data only.

## Outcome summary

{md_table(rows, ['set', 'PII recall', 'doc macro-F1', 'field F1', 'route HV recall', 'route suspicious recall', 'route ECE', 'auto coverage'])}

Real OCR PII recall covers only identifiers that survived OCR verbatim; Tesseract changes offsets, so the remaining identifiers cannot be aligned reliably. The hard set uses held-out layouts, heavy literal noise, and simulated OCR.

Golden routing recall has limited support: high-value uses {golden_route['high_value']['support']} documents with a 95% Wilson interval of {golden_route['high_value']['recall_ci95']}; suspicious uses {golden_route['suspicious']['support']} documents with an interval of {golden_route['suspicious']['recall_ci95']}.

## Golden-set target check

{md_table(target_rows, ['metric', 'target', 'observed', 'status'])}

## Confidence gate

The gate threshold was selected on validation data to require at least 98% joint document-type and route accuracy. Golden-set coverage is the auto-triage estimate; rejected documents go to human review.

## Limits

These are synthetic-data baseline results. The field extractor learns value dictionaries from the training split, and routing labels were generated by rules. A real insurer sample and a trained token model are still required before production claims can be made. The current gate covers classification and routing confidence; rule-based PII and field stages do not yet emit calibrated probabilities.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus_synthea")
    parser.add_argument("--control-corpus", default="data/corpus")
    parser.add_argument("--output", default="reports/model_results.json")
    parser.add_argument("--report", default="docs/model_evaluation.md")
    parser.add_argument("--models", default="artifacts/models")
    args = parser.parse_args()

    corpus = ROOT / args.corpus
    datasets = {name: load_jsonl(corpus / f"{name}.jsonl") for name in ("train", "val", "test_golden", "test_hard", "real_ocr")}
    control_corpus = ROOT / args.control_corpus
    control = {name: load_jsonl(control_corpus / f"{name}.jsonl") for name in ("test_golden", "test_hard", "real_ocr")}
    train, val = datasets["train"], datasets["val"]

    doc_model, doc_experiments = fit_best(train, val, "doc_type")
    redactor = Redactor()
    fields = FieldExtractor().fit(train)
    route_model, route_experiments = fit_route(train, val, fields, doc_model)

    val_doc_metrics, val_doc_pred, val_doc_conf = evaluate_classifier(doc_model, val, "doc_type")
    val_route_metrics, val_route_pred, val_route_conf = evaluate_classifier(route_model, val, "route")
    gate = confidence_gate(
        np.asarray([doc["doc_type"] for doc in val]) == val_doc_pred,
        np.asarray([doc["route"] for doc in val]) == val_route_pred,
        val_doc_conf,
        val_route_conf,
    )

    evaluation = {}
    timings = []
    evaluation_sets = {name: datasets[name] for name in ("test_golden", "test_hard", "real_ocr")}
    evaluation_sets.update({f"control_{name}": documents for name, documents in control.items()})
    for split, documents in evaluation_sets.items():
        doc_score, doc_pred, doc_conf = evaluate_classifier(doc_model, documents, "doc_type")
        route_score, route_pred, route_conf = evaluate_classifier(route_model, documents, "route")
        pii_predictions = [redactor.find(doc["text"]) for doc in documents]
        pii_score = pii_string_metrics(documents, pii_predictions) if split.endswith("real_ocr") else pii_metrics(documents, pii_predictions)
        field_predictions = fields.predict(documents, doc_pred)
        field_score = field_metrics(documents, field_predictions)
        gate_score = apply_gate(
            gate,
            [doc["doc_type"] for doc in documents], doc_pred, doc_conf,
            [doc["route"] for doc in documents], route_pred, route_conf,
        )
        evaluation[split] = {
            "pii": pii_score,
            "document_type": doc_score,
            "fields": field_score,
            "routing": route_score,
            "gate": gate_score,
        }

    for doc in datasets["test_golden"][:150]:
        started = time.perf_counter()
        redactor.find(doc["text"])
        predicted_type = doc_model.predict([doc["text"]])[0]
        fields.predict_one(doc["text"], predicted_type)
        route_model.predict([doc["text"]])
        timings.append((time.perf_counter() - started) * 1000)
    latency = {"p50_ms": round(float(np.percentile(timings, 50)), 2), "p95_ms": round(float(np.percentile(timings, 95)), 2)}

    golden = evaluation["test_golden"]
    target_status = {
        "PII leak recall": {"target": ">= 0.97", "observed": golden["pii"]["leak_recall"], "pass": golden["pii"]["leak_recall"] >= 0.97},
        "Document macro-F1": {"target": ">= 0.95", "observed": golden["document_type"]["macro_f1"], "pass": golden["document_type"]["macro_f1"] >= 0.95},
        "Field F1": {"target": ">= 0.90", "observed": golden["fields"]["f1"], "pass": golden["fields"]["f1"] >= 0.90},
        "High-value route recall": {"target": ">= 0.90", "observed": golden["routing"]["per_class"]["high_value"]["recall"], "pass": golden["routing"]["per_class"]["high_value"]["recall"] >= 0.90},
        "Suspicious route recall": {"target": ">= 0.90", "observed": golden["routing"]["per_class"]["suspicious"]["recall"], "pass": golden["routing"]["per_class"]["suspicious"]["recall"] >= 0.90},
        "Document ECE": {"target": "<= 0.05", "observed": golden["document_type"]["ece_15"], "pass": golden["document_type"]["ece_15"] <= 0.05},
        "Routing ECE": {"target": "<= 0.05", "observed": golden["routing"]["ece_15"], "pass": golden["routing"]["ece_15"] <= 0.05},
        "p95 CPU latency": {"target": "< 2000 ms", "observed": latency["p95_ms"], "pass": latency["p95_ms"] < 2000},
        "Auto-triage coverage": {"target": ">= 0.70", "observed": golden["gate"]["coverage"], "pass": golden["gate"]["coverage"] >= 0.70},
    }

    manifest = json.loads((corpus / "manifest.json").read_text())
    result = {
        "run": {"python": platform.python_version(), "sklearn": sklearn.__version__, "platform": platform.platform()},
        "data": {
            "corpus_version": manifest["corpus_version"],
            "control_corpus_version": json.loads((control_corpus / "manifest.json").read_text())["corpus_version"],
            "train_documents": len(train),
            "validation_documents": len(val),
            "file_sha256": {name: sha256(corpus / f"{name}.jsonl") for name in datasets},
            "control_file_sha256": {name: sha256(control_corpus / f"{name}.jsonl") for name in control},
        },
        "experiments": {"document_type": doc_experiments, "routing": route_experiments},
        "validation": {"document_type": val_doc_metrics, "routing": val_route_metrics, "gate": gate},
        "evaluation": evaluation,
        "latency": latency,
        "target_status": target_status,
    }

    model_dir = ROOT / args.models
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "document_type": doc_model,
            "routing": route_model,
            "fields": fields,
            "redactor": redactor,
            "confidence_gate": gate,
            "corpus_version": manifest["corpus_version"],
        },
        model_dir / "baseline_bundle.joblib",
    )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    report = ROOT / args.report
    report.write_text(make_report(result))
    print(json.dumps({"results": str(output), "report": str(report), "models": str(model_dir), "targets": target_status}, indent=2))


if __name__ == "__main__":
    main()
