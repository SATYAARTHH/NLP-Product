"""High-recall, context-aware rule baseline for direct identifiers."""

from __future__ import annotations

import re


class Redactor:
    FLAGS = re.IGNORECASE | re.MULTILINE

    def __init__(self):
        date = r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{1,2}-[A-Za-z]{3}-\d{4}|[A-Za-z]+\s+\d{1,2},\s+\d{4})"
        self.patterns = [
            ("EMAIL", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", self.FLAGS)),
            ("SSN", re.compile(r"(?<!\d)(?:9\d{2}[- ]?\d{2}[- ]?\d{4})(?!\d)", self.FLAGS)),
            ("PHONE", re.compile(r"(?<!\d)(?:\(\d{3}\)\s*|\d{3}[-. ]?)\d{3}[-. ]?\d{4}(?!\d)", self.FLAGS)),
            ("MEMBER_ID", re.compile(r"\bMB(?:[- ]?\d){7}\b", self.FLAGS)),
            ("MRN", re.compile(r"\bMRN[- ]?\d{8}\b", self.FLAGS)),
            ("ACCOUNT_NUMBER", re.compile(r"\b(?:AC[- ]?\d{9}|\d{3}-\d{3}-\d{3})\b", self.FLAGS)),
            ("DATE_OF_BIRTH", re.compile(r"(?:DOB|D[.]?O[.]?B[.]?|Date\s+of\s+Birth|born)\s*[:.]?\s*(?P<value>" + date + r")", self.FLAGS)),
            ("ADDRESS", re.compile(r"(?:Address|Addr[.]?|residing\s+at)\s*[:.]?\s*(?P<value>[^\n|]+?)(?=\.?\n|\.\s*(?:Policy|Member)|$)", self.FLAGS)),
        ]
        name_prefix = (
            r"(?:Patient|Pt[.]?|Name|Insured|Re|Attending[ \t]+physician|Ordering[ \t]+physician|"
            r"Prescriber|Requesting[ \t]+physician|Signed|Family[ \t]+contact|Electronically[ \t]+signed[ \t]+by)"
        )
        self.name_pattern = re.compile(
            r"^[ \t]*" + name_prefix
            + r"[ \t]*(?:[.:]+|\.{2,})?[ \t]*"
            + r"(?P<value>(?:Dr[.]?[ \t]+)?(?:[A-Z][A-Za-z'-]*[.]?(?:,[ \t]*|[ \t]+)){1,2}"
            + r"[A-Z][A-Za-z'-]*(?:,[ \t]*MD)?)",
            self.FLAGS,
        )
        self.sincerely_pattern = re.compile(
            r"(?:^Sincerely,?[ \t]*\n|^Electronically[ \t]+signed[ \t]+by[ \t]+)"
            r"(?P<value>(?:Dr[.]?[ \t]+)?(?:[A-Z][A-Za-z'-]*[.]?(?:,[ \t]*|[ \t]+)){1,2}"
            r"[A-Z][A-Za-z'-]*(?:,[ \t]*MD)?)",
            self.FLAGS,
        )
        self.mrn_context = re.compile(r"(?:MRN|Medical\s+Record\s+Number|Med\s+Rec\s+No)\s*[:.]?\s*(?P<value>\d{8})", self.FLAGS)

    @staticmethod
    def _span(label: str, match: re.Match) -> dict:
        group = "value" if "value" in match.re.groupindex else 0
        start, end = match.span(group)
        return {"label": label, "start": start, "end": end, "text": match.string[start:end]}

    def find(self, text: str) -> list[dict]:
        found = []
        for label, pattern in self.patterns:
            found.extend(self._span(label, match) for match in pattern.finditer(text))
        found.extend(self._span("NAME", match) for match in self.name_pattern.finditer(text))
        found.extend(self._span("NAME", match) for match in self.sincerely_pattern.finditer(text))
        found.extend(self._span("MRN", match) for match in self.mrn_context.finditer(text))
        # Patient names often repeat later in a narrative without another label.
        for item in list(found):
            if item["label"] != "NAME" or len(item["text"]) < 4:
                continue
            for match in re.finditer(re.escape(item["text"]), text, re.IGNORECASE):
                found.append({"label": "NAME", "start": match.start(), "end": match.end(), "text": match.group()})
        unique = {(item["start"], item["end"], item["label"]): item for item in found}
        return sorted(unique.values(), key=lambda item: (item["start"], item["end"]))

    def redact(self, text: str) -> tuple[str, list[dict]]:
        spans = self.find(text)
        safe = text
        for span in sorted(spans, key=lambda item: item["start"], reverse=True):
            safe = safe[:span["start"]] + f"[{span['label']}]" + safe[span["end"]:]
        return safe, spans
