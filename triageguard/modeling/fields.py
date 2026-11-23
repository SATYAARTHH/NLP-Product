"""Anchor and training-lexicon baseline for normalized field extraction."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime

from triageguard.schema import FIELDS_BY_TYPE


DATE_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|"
    r"\d{1,2}-[A-Za-z]{3}-\d{4}|[A-Za-z]+\s+\d{1,2},\s+\d{4}", re.I
)
MONEY_RE = re.compile(r"(?:USD\s*|\$\s*)?\d{1,3}(?:,\d{3})*(?:\.\d{2})|(?:USD\s*|\$\s*)?\d+\.\d{2}", re.I)


def _date(value: str) -> str | None:
    value = value.strip()
    formats = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%m.%d.%Y", "%m.%d.%y", "%d-%b-%Y", "%B %d, %Y")
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def _money(value: str) -> float | None:
    try:
        return round(float(re.sub(r"[^0-9.]", "", value)), 2)
    except ValueError:
        return None


class FieldExtractor:
    """Learns allowed text values, while dates and amounts remain rule based."""

    def fit(self, documents: list[dict]):
        self.values = defaultdict(set)
        self.composites = defaultdict(list)
        self.aliases = defaultdict(dict)
        for doc in documents:
            for field, value in doc["fields"].items():
                if value is None:
                    continue
                if isinstance(value, str) and not field.endswith("date"):
                    self.values[field].add(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            self.values[field].add(item)
                        else:
                            self.composites[field].append(item)
            for span in doc.get("field_spans", []):
                field = span["field"]
                normalized = doc["fields"].get(field)
                if isinstance(normalized, str):
                    self.aliases[field].setdefault(normalized, set()).add(span["text"])
        return self

    @staticmethod
    def _after(text: str, anchors: str, pattern: re.Pattern) -> str | None:
        match = re.search(anchors, text, re.I)
        if not match:
            return None
        found = pattern.search(text, match.end(), min(len(text), match.end() + 120))
        return found.group(0) if found else None

    def _lexicon(self, field: str, text: str, multiple: bool = False):
        lower = text.casefold()
        matches = []
        for value in sorted(self.values[field]):
            aliases = self.aliases[field].get(value, {value}) | {value}
            for alias in sorted(aliases):
                position = lower.find(alias.casefold())
                if position >= 0:
                    matches.append((position, position + len(alias), value))
        matches.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
        values = []
        spans = []
        for start, end, value in matches:
            if value not in values and not any(start >= lo and end <= hi for lo, hi in spans):
                values.append(value)
                spans.append((start, end))
        return values if multiple else (values[0] if values else None)

    def _date_field(self, field: str, text: str) -> str | None:
        anchors = {
            "service_date": r"(?:Date\s+of\s+service|DOS|treated\s+at[^\n]+?\s+on)\s*[:/]?\s*",
            "admission_date": r"(?:Admission\s+Date|Admitted|ADM)\s*[:/]?\s*",
            "discharge_date": r"(?:Discharge\s+Date|Discharged|D/C|DC)\s*[:/]?\s*",
            "collection_date": r"(?:Collected|collection\s+date|drawn\s+on|COLL\s+DT)\s*[:/]?\s*",
            "prescription_date": r"Date\s*:\s*",
        }
        raw = self._after(text, anchors[field], DATE_RE)
        return _date(raw) if raw else None

    def _money_field(self, field: str, text: str) -> float | None:
        anchors = {
            "claimed_amount": r"(?:amount\s+claimed|Amt[.]?\s+claimed|TOTAL\s+CLAIMED|Sum\s+requested)",
            "total_billed": r"(?:Total\s+billed|TOTAL\s+CHARGES|Amount\s+due|GRAND\s+TOTAL)",
            "estimated_cost": r"(?:estimated\s+cost|Estimated\s+charges|Est[.]?\s*\$)",
        }
        raw = self._after(text, anchors[field], MONEY_RE)
        return _money(raw) if raw else None

    def _line_items(self, text: str) -> list[dict]:
        result = []
        lower = text.casefold()
        candidates = {item["description"] for item in self.composites["line_items"]}
        occurrences = []
        for description in sorted(candidates):
            start = lower.find(description.casefold())
            if start >= 0:
                occurrences.append((start, start + len(description), description))
        selected = []
        for start, end, description in sorted(occurrences, key=lambda item: (item[0], -(item[1] - item[0]), item[2])):
            if any(start >= lo and end <= hi for lo, hi in selected):
                continue
            selected.append((start, end))
            line = text[start:text.find("\n", start) if "\n" in text[start:] else len(text)]
            amounts = MONEY_RE.findall(line)
            if amounts:
                result.append((start, {"description": description, "amount": _money(amounts[-1])}))
        return [item for _, item in sorted(result, key=lambda pair: pair[0])]

    def _tests(self, text: str) -> list[dict]:
        result = []
        lower = text.casefold()
        by_name = {}
        for item in self.composites["tests"]:
            by_name.setdefault(item["name"], item["unit"])
        occurrences = []
        for name, unit in sorted(by_name.items()):
            start = lower.find(name.casefold())
            if start >= 0:
                occurrences.append((start, start + len(name), name, unit))
        selected = []
        for start, end, name, unit in sorted(occurrences, key=lambda item: (item[0], -(item[1] - item[0]), item[2])):
            if any(start >= lo and end <= hi for lo, hi in selected):
                continue
            selected.append((start, end))
            tail = text[start + len(name):start + len(name) + 100]
            value_match = re.search(r"[-+]?\d+(?:\.\d+)?", tail)
            if value_match:
                result.append((start, {"name": name, "value": float(value_match.group()), "unit": unit}))
        return [item for _, item in sorted(result, key=lambda pair: pair[0])]

    def _medicines(self, text: str) -> list[dict]:
        result = []
        lower = text.casefold()
        unique = {(item["name"], item["dose"]) for item in self.composites["medicines"]}
        for name, dose in sorted(unique):
            start = lower.find(name.casefold())
            if start >= 0 and dose.casefold() in lower[start:start + len(name) + 50]:
                result.append((start, {"name": name, "dose": dose}))
        seen = set()
        ordered = []
        for _, item in sorted(result, key=lambda pair: pair[0]):
            key = (item["name"], item["dose"])
            if key not in seen:
                ordered.append(item)
                seen.add(key)
        return ordered

    def predict_one(self, text: str, doc_type: str) -> dict:
        prediction = {}
        list_text = {"procedure", "procedures"}
        dates = {"service_date", "admission_date", "discharge_date", "collection_date", "prescription_date"}
        amounts = {"claimed_amount", "total_billed", "estimated_cost"}
        for field in FIELDS_BY_TYPE[doc_type]:
            if field in dates:
                prediction[field] = self._date_field(field, text)
            elif field in amounts:
                prediction[field] = self._money_field(field, text)
            elif field in list_text:
                prediction[field] = self._lexicon(field, text, multiple=True)
            elif field == "line_items":
                prediction[field] = self._line_items(text)
            elif field == "tests":
                prediction[field] = self._tests(text)
            elif field == "medicines":
                prediction[field] = self._medicines(text)
            else:
                prediction[field] = self._lexicon(field, text)
        return prediction

    def predict(self, documents: list[dict], doc_types=None) -> list[dict]:
        if doc_types is None:
            doc_types = [doc["doc_type"] for doc in documents]
        return [self.predict_one(doc["text"], doc_type) for doc, doc_type in zip(documents, doc_types)]
