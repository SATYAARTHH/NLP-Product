"""Metrics fixed before model training, following the M1 decision log."""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score


def expected_calibration_error(y_true, probabilities, classes, n_bins: int = 15) -> float:
    """Top-class ECE with equal-width bins."""
    probabilities = np.asarray(probabilities)
    classes = np.asarray(classes)
    confidence = probabilities.max(axis=1)
    predicted = classes[probabilities.argmax(axis=1)]
    correct = predicted == np.asarray(y_true)
    total = len(y_true)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (confidence > low) & (confidence <= high)
        if selected.any():
            ece += selected.sum() / total * abs(correct[selected].mean() - confidence[selected].mean())
    return float(ece)


def classification_metrics(y_true, y_pred, probabilities, classes) -> dict:
    report = classification_report(y_true, y_pred, labels=list(classes), output_dict=True, zero_division=0)
    per_class = {}
    truth = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    for name in classes:
        values = {key: round(float(report[name][key]), 4) for key in ("precision", "recall", "f1-score")}
        support = int((truth == name).sum())
        hits = int(((truth == name) & (predicted == name)).sum())
        if support:
            z = 1.96
            proportion = hits / support
            denominator = 1 + z * z / support
            centre = (proportion + z * z / (2 * support)) / denominator
            margin = z * math.sqrt(proportion * (1 - proportion) / support + z * z / (4 * support * support)) / denominator
            values["recall_ci95"] = [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]
        values["support"] = support
        per_class[name] = values
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "ece_15": round(expected_calibration_error(y_true, probabilities, classes), 4),
        "per_class": per_class,
    }


def pii_metrics(documents: list[dict], predictions: list[list[dict]]) -> dict:
    """Leak recall requires the full gold identifier to be covered."""
    total = leaked = exact_hits = 0
    predicted_total = 0
    by_label: dict[str, Counter] = {}
    for doc, predicted in zip(documents, predictions):
        predicted_total += len(predicted)
        exact = {(p["label"], p["start"], p["end"]) for p in predicted}
        for gold in doc.get("pii", []):
            total += 1
            label = gold["label"]
            stats = by_label.setdefault(label, Counter())
            stats["total"] += 1
            covered = [False] * (gold["end"] - gold["start"])
            for pred in predicted:
                lo = max(gold["start"], pred["start"])
                hi = min(gold["end"], pred["end"])
                for i in range(lo, hi):
                    covered[i - gold["start"]] = True
            if covered and all(covered):
                leaked += 1
                stats["covered"] += 1
            if (label, gold["start"], gold["end"]) in exact:
                exact_hits += 1
                stats["exact"] += 1
    label_result = {
        label: {
            "leak_recall": round(v["covered"] / v["total"], 4),
            "exact_recall": round(v["exact"] / v["total"], 4),
            "support": v["total"],
        }
        for label, v in sorted(by_label.items())
    }
    return {
        "leak_recall": round(leaked / total, 4) if total else 0.0,
        "exact_recall": round(exact_hits / total, 4) if total else 0.0,
        "exact_precision": round(exact_hits / predicted_total, 4) if predicted_total else 0.0,
        "entities": total,
        "per_label": label_result,
    }


def pii_string_metrics(documents: list[dict], predictions: list[list[dict]]) -> dict:
    """Score identifiers that survived real OCR and can still be located exactly."""
    total = covered_total = 0
    for doc, predicted in zip(documents, predictions):
        lower = doc["text"].casefold()
        for gold in doc.get("pii_strings", []):
            needle = gold["text"].casefold()
            start = lower.find(needle)
            if start < 0:
                continue
            total += 1
            end = start + len(needle)
            covered = [False] * len(needle)
            for pred in predicted:
                lo = max(start, pred["start"])
                hi = min(end, pred["end"])
                for i in range(lo, hi):
                    covered[i - start] = True
            if covered and all(covered):
                covered_total += 1
    return {
        "surviving_string_recall": round(covered_total / total, 4) if total else 0.0,
        "surviving_entities": total,
        "note": "Only identifiers still present verbatim after Tesseract are scorable without aligned spans.",
    }


def _canonical(value) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    if isinstance(value, dict):
        return json.dumps({k: _canonical(v) for k, v in sorted(value.items())}, sort_keys=True)
    return str(value).casefold()


def field_instances(fields: dict) -> Counter:
    """Turn scalar fields and each list element into independently scored instances."""
    instances = Counter()
    for field, value in fields.items():
        if value is None:
            continue
        values: Iterable = value if isinstance(value, list) else [value]
        for item in values:
            instances[(field, _canonical(item))] += 1
    return instances


def field_metrics(documents: list[dict], predictions: list[dict]) -> dict:
    tp = fp = fn = 0
    by_type: dict[str, Counter] = {}
    for doc, predicted in zip(documents, predictions):
        gold = field_instances(doc["fields"])
        guess = field_instances(predicted)
        hits = gold & guess
        local_tp = sum(hits.values())
        local_fp = sum((guess - gold).values())
        local_fn = sum((gold - guess).values())
        tp += local_tp
        fp += local_fp
        fn += local_fn
        c = by_type.setdefault(doc["doc_type"], Counter())
        c.update(tp=local_tp, fp=local_fp, fn=local_fn)

    def scores(c: Counter) -> dict:
        precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    overall = scores(Counter(tp=tp, fp=fp, fn=fn))
    overall["per_doc_type"] = {name: scores(counts) for name, counts in sorted(by_type.items())}
    return overall


def confidence_gate(doc_correct, route_correct, doc_confidence, route_confidence,
                    required_accuracy: float = 0.98) -> dict:
    """Choose the lowest validation threshold meeting joint accuracy."""
    joint_correct = np.asarray(doc_correct) & np.asarray(route_correct)
    confidence = np.minimum(doc_confidence, route_confidence)
    best = {"threshold": 1.0, "coverage": 0.0, "accuracy": 1.0}
    for threshold in np.linspace(0.0, 1.0, 201):
        accepted = confidence >= threshold
        if not accepted.any():
            continue
        accuracy = joint_correct[accepted].mean()
        coverage = accepted.mean()
        if accuracy >= required_accuracy and coverage > best["coverage"]:
            best = {"threshold": float(threshold), "coverage": float(coverage), "accuracy": float(accuracy)}
    return {key: round(value, 4) for key, value in best.items()}
