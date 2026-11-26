"""Small local demo service around the frozen M3/M4 model bundle."""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path

import joblib
from docx import Document
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pypdf import PdfReader

from triageguard.data.clean import clean_text
from triageguard.modeling.redaction import Redactor
from triageguard.schema import DOC_TYPES

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("TRIAGEGUARD_MODEL", ROOT / "artifacts/models/baseline_bundle.joblib"))
MAX_BYTES = 5 * 1024 * 1024
MAX_CHARS = 100_000
logger = logging.getLogger("triageguard.api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

app = FastAPI(title="TriageGuard local prototype", version="m5-minimal")
_bundle = None


def bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.is_file():
            raise RuntimeError(f"Model bundle missing: {MODEL_PATH}. Run scripts/train_evaluate.py first.")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def extract_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".txt":
            text = content.decode("utf-8-sig")
        elif suffix == ".pdf":
            reader = PdfReader(io.BytesIO(content))
            if len(reader.pages) > 20:
                raise HTTPException(413, "PDF has more than 20 pages")
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif suffix == ".docx":
            document = Document(io.BytesIO(content))
            parts = [paragraph.text for paragraph in document.paragraphs]
            parts += [cell.text for table in document.tables for row in table.rows for cell in row.cells]
            text = "\n".join(parts)
        else:
            raise HTTPException(415, "Upload a .txt, .docx, or text-layer .pdf file")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, "Could not read this document") from exc
    if not text.strip():
        raise HTTPException(422, "No readable text found. Scanned PDFs need OCR before upload.")
    if len(text) > MAX_CHARS:
        raise HTTPException(413, "Extracted text exceeds 100,000 characters")
    return clean_text(text)


def classify(model, text: str) -> tuple[str, float]:
    scores = model.predict_proba([text])[0]
    index = int(scores.argmax())
    return str(model.classes_[index]), float(scores[index])


def triage(text: str) -> dict:
    start = time.perf_counter()
    models = bundle()
    safe_text, _ = models.get("redactor", Redactor()).redact(text)
    fallback = False
    try:
        doc_type, doc_confidence = classify(models["document_type"], text)
    except Exception:
        logger.warning("document classifier failed")
        # A classifier error cannot be safely auto-routed.
        doc_type, doc_confidence, fallback = "unknown", 0.0, True
    try:
        fields = models["fields"].predict_one(text, doc_type) if doc_type in DOC_TYPES else {}
        # Doctor names are Tier 1 identifiers, even when extracted as fields.
        if fields.get("doctor"):
            fields["doctor"] = "[NAME]"
    except Exception:
        logger.warning("field extraction failed")
        fields, fallback = {}, True
    try:
        route, route_confidence = classify(models["routing"], text)
    except Exception:
        logger.warning("route classifier failed")
        fallback = True
        try:
            rules = models["routing"].estimator.rules
            route, _ = rules.predict_one(text, doc_type if doc_type in DOC_TYPES else None)
        except Exception:
            route = "routine"
        route_confidence = 0.0
    try:
        estimator = models["routing"].estimator
        rules = getattr(estimator, "rules", estimator)
        route_signals = rules.detect_signals(text, doc_type) if doc_type in DOC_TYPES and hasattr(rules, "detect_signals") else []
    except Exception:
        route_signals = []
    threshold = float(models["confidence_gate"]["threshold"])
    needs_review = fallback or min(doc_confidence, route_confidence) < threshold
    return {
        "safe_text": safe_text,
        "doc_type": doc_type,
        "doc_type_confidence": round(doc_confidence, 4),
        "fields": fields,
        "route": route,
        "route_confidence": round(route_confidence, 4),
        "route_signals": route_signals,
        "needs_review": needs_review,
        "review_reason": "model_fallback" if fallback else ("low_confidence" if needs_review else None),
        "gate_threshold": threshold,
        "model_version": models["corpus_version"],
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }


@app.get("/health")
def health():
    try:
        models = bundle()
    except Exception:
        raise HTTPException(503, "Model bundle unavailable")
    return {"status": "ok", "model_version": models["corpus_version"]}


@app.post("/triage")
async def triage_upload(file: UploadFile = File(...)):
    if Path(file.filename or "").suffix.lower() not in {".txt", ".pdf", ".docx"}:
        raise HTTPException(415, "Upload a .txt, .docx, or text-layer .pdf file")
    content = await file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(413, "File exceeds 5 MB")
    text = extract_text(file.filename or "", content)
    try:
        result = triage(text)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    # Structured monitoring contains no document text, identifiers, or extracted values.
    logger.info("triage type=%s route=%s review=%s latency_ms=%s", result["doc_type"], result["route"], result["needs_review"], result["latency_ms"])
    return result


@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).parent / "static" / "index.html").read_text()
