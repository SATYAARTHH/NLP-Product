"""Visible-fact routing baseline mirroring the documented business rules."""

from __future__ import annotations

import re
from datetime import datetime

import numpy as np

from triageguard.data.routing import AMBIGUOUS_FLOOR, HIGH_VALUE, LONG_STAY_DAYS
from triageguard.modeling.fields import DATE_RE, FieldExtractor, _date


class RoutingRules:
    classes_ = np.asarray(["high_value", "routine", "suspicious"])

    def __init__(self, fields: FieldExtractor, document_model):
        self.fields = fields
        self.document_model = document_model

    def fit(self, documents: list[dict]):
        self.compatible = {}
        self.incompatible = set()
        self.known_procedures = set()
        for doc in documents:
            diagnosis = doc["fields"].get("diagnosis")
            procedures = []
            for key in ("procedure", "procedures"):
                procedures.extend(doc["fields"].get(key) or [])
            requested = doc["fields"].get("requested_procedure")
            if requested:
                procedures.append(requested)
            self.known_procedures.update(procedures)
            if diagnosis:
                if "procedure_mismatch" in doc.get("route_signals", []):
                    self.incompatible.update((diagnosis, procedure) for procedure in procedures)
                else:
                    self.compatible.setdefault(diagnosis, set()).update(procedures)
        return self

    @staticmethod
    def _dob(text: str):
        match = re.search(r"(?:DOB|D[.]?O[.]?B[.]?|Date\s+of\s+Birth|born)\s*[:.]?\s*(" + DATE_RE.pattern + r")", text, re.I)
        value = _date(match.group(1)) if match else None
        if not value:
            return None
        parsed = datetime.fromisoformat(value).date()
        if parsed.year > 2026:
            parsed = parsed.replace(year=parsed.year - 100)
        return parsed

    def _duplicate(self, text: str) -> bool:
        lower = text.casefold()
        descriptions = {item["description"] for item in self.fields.composites["line_items"]}
        return any(lower.count(description.casefold()) >= 2 for description in descriptions)

    def _mismatch(self, text: str) -> bool:
        diagnosis = self.fields._lexicon("diagnosis", text)
        if not diagnosis or diagnosis not in self.compatible:
            return False
        procedures = []
        for field in ("procedure", "procedures", "requested_procedure"):
            value = self.fields._lexicon(field, text, multiple=field != "requested_procedure")
            procedures.extend(value if isinstance(value, list) else ([value] if value else []))
        procedures.extend(
            item["description"] for item in self.fields._line_items(text)
            if item["description"] in self.known_procedures
        )
        allowed = self.compatible.get(diagnosis, set())
        return any((diagnosis, procedure) in self.incompatible and procedure not in allowed for procedure in procedures if procedure)

    def predict_one(self, text: str, doc_type: str | None = None) -> tuple[str, float]:
        if doc_type is None:
            doc_type = self.document_model.predict([text])[0]
        extracted = self.fields.predict_one(text, doc_type)
        dates = [value for value in DATE_RE.findall(text) if _date(value)]
        dob = self._dob(text)
        parsed_dates = [datetime.fromisoformat(_date(value)).date() for value in dates]
        admission = extracted.get("admission_date")
        discharge = extracted.get("discharge_date")
        impossible = bool(
            admission and discharge and datetime.fromisoformat(discharge) < datetime.fromisoformat(admission)
        ) or bool(dob and any(value < dob for value in parsed_dates if value != dob))
        if self._duplicate(text) or impossible or self._mismatch(text):
            return "suspicious", 0.98

        amount = extracted.get("claimed_amount") or extracted.get("total_billed") or extracted.get("estimated_cost")
        if amount is not None:
            high = amount >= HIGH_VALUE
            if not high and amount >= AMBIGUOUS_FLOOR:
                long_stay = False
                if admission and discharge:
                    delta = datetime.fromisoformat(discharge) - datetime.fromisoformat(admission)
                    long_stay = delta.days >= LONG_STAY_DAYS
                procedures = extracted.get("procedure") or extracted.get("procedures") or []
                items = extracted.get("line_items") or []
                high = long_stay or len(procedures) >= 2 or len(items) >= 4
            if high:
                return "high_value", 0.98
        return "routine", 0.98

    def predict_proba(self, texts):
        probabilities = []
        for item in texts:
            text, doc_type = item if isinstance(item, tuple) else (item, None)
            label, confidence = self.predict_one(text, doc_type)
            row = np.full(len(self.classes_), (1.0 - confidence) / (len(self.classes_) - 1))
            row[np.where(self.classes_ == label)[0][0]] = confidence
            probabilities.append(row)
        return np.asarray(probabilities)

    def predict(self, texts):
        probabilities = self.predict_proba(texts)
        return self.classes_[probabilities.argmax(axis=1)]


class HybridRouting:
    """Union safety signals from rules and text; agreement supplies confidence."""

    classes_ = RoutingRules.classes_

    def __init__(self, rules: RoutingRules, learned_model):
        self.rules = rules
        self.learned_model = learned_model

    def predict_proba(self, texts):
        rule_probabilities = self.rules.predict_proba(texts)
        learned_probabilities = self.learned_model.predict_proba(texts)
        rule_prediction = self.classes_[rule_probabilities.argmax(axis=1)]
        learned_prediction = self.classes_[learned_probabilities.argmax(axis=1)]
        output = []
        for rule, learned, learned_row in zip(rule_prediction, learned_prediction, learned_probabilities):
            selected = "suspicious" if "suspicious" in (rule, learned) else rule
            confidence = 0.99 if rule == learned else 0.60
            selected_index = int(np.where(self.classes_ == selected)[0][0])
            row = np.asarray(learned_row, dtype=float).copy()
            other_total = row.sum() - row[selected_index]
            if other_total <= 0:
                row[:] = (1.0 - confidence) / (len(row) - 1)
            else:
                row *= (1.0 - confidence) / other_total
            row[selected_index] = confidence
            output.append(row)
        return np.asarray(output)

    def predict(self, texts):
        probabilities = self.predict_proba(texts)
        return self.classes_[probabilities.argmax(axis=1)]
