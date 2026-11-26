# M5/M6 minimal delivery and live demo

## What is delivered

The M5 prototype wraps the existing frozen M3/M4 bundle in a FastAPI service. `POST /triage` accepts one UTF-8 `.txt`, `.docx`, or text-layer `.pdf` file and returns redacted text, one of six document types, type confidence, extracted fields, a routine/high-value/suspicious route, route confidence, a review decision, the validation-selected gate threshold, model version, and latency. `GET /health` checks the bundle. The home page at `/` is a small upload UI; `/docs` is FastAPI's API explorer.

Files are limited to 5 MB, PDFs to 20 pages, and extracted text to 100,000 characters. An image-only PDF returns a clear error because this lean service has no OCR adapter. The model was trained on synthetic US insurance paperwork; outputs on arbitrary real documents need human review. Field rules have no calibrated confidence, so the gate covers document type and routing only. `route_signals` lists visible rule findings; a text-model warning can still have no named rule signal.

The service keeps uploaded bytes and raw text in memory for the request. Its normal log line contains only predicted type, route, review status, and latency. It masks the doctor field before returning it. The frozen bundle is loaded from `artifacts/models/baseline_bundle.joblib`; `TRIAGEGUARD_MODEL` can point to a different trusted, compatible bundle for a simple rollback. Only load bundles you trust because joblib uses Python pickle.

## Run locally

In a terminal at the repository root:

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
test -f artifacts/models/baseline_bundle.joblib || .venv/bin/python scripts/train_evaluate.py
.venv/bin/python -m uvicorn triageguard.api:app --host 127.0.0.1 --port 8000
```

Leave that terminal open. In a second terminal, check readiness and create an upload from the first frozen **synthetic** golden document:

```bash
curl -s http://127.0.0.1:8000/health
.venv/bin/python -c 'import json,pathlib; d=json.loads(pathlib.Path("data/corpus_synthea/test_golden.jsonl").open().readline()); pathlib.Path("/tmp/triageguard-demo.txt").write_text(d["text"]); print(d["doc_type"], d["route"])'
curl -s -F 'file=@/tmp/triageguard-demo.txt' http://127.0.0.1:8000/triage | .venv/bin/python -m json.tool
```

Expected for that example: `claim_form`, `routine`, `needs_review: false`, and placeholders such as `[NAME]` and `[SSN]` in `safe_text`. The sample's source label is printed by the preparation command. To use the browser, open <http://127.0.0.1:8000>, select `/tmp/triageguard-demo.txt`, and click **Run triage**. To try a document of your own, upload a `.txt`, `.docx`, or searchable PDF through the same page. Confirm the redacted copy, fields, route, review flag, and latency. Stop the server with Ctrl-C.

For a DOCX trial from the same sample:

```bash
.venv/bin/python -c 'from docx import Document; from pathlib import Path; d=Document(); [d.add_paragraph(line) for line in Path("/tmp/triageguard-demo.txt").read_text().splitlines()]; d.save("/tmp/triageguard-demo.docx")'
curl -s -F 'file=@/tmp/triageguard-demo.docx' http://127.0.0.1:8000/triage | .venv/bin/python -m json.tool
```

## Docker option

After the bundle exists, run:

```bash
docker build -t triageguard-demo .
docker run --rm -p 8000:8000 triageguard-demo
```

Then use the same browser or `curl` steps. The Docker build copies the locally generated bundle into the image.

## M6 report and presentation outline

**Problem.** Claim paperwork needs first-pass sorting while direct identifiers remain protected. The baseline handles six document types and three routes; “suspicious” is a review signal, not a fraud finding.

**Method.** A synthetic Synthea/Faker corpus trains a word TF-IDF document classifier, rule/lexicon field extraction, rule-based redaction, and hybrid routing. A validation-selected threshold sends uncertain type/route predictions to human review. The M5 adapter extracts text from simple uploads and returns the M4 model output.

**Measured result.** On the frozen synthetic golden set: PII recall 0.9841, document macro-F1 1.0000, field F1 0.9684, high-value recall 0.9744, suspicious recall 0.9400, and auto-triage coverage 0.8115. The M4 CPU p95 was 2.84 ms for model stages; this is **not** an API or PDF-extraction benchmark. Hard/OCR and cross-source scores fall substantially; see [model evaluation](model_evaluation.md).

**Live presentation (four short slides).** 1. Problem and privacy boundary. 2. Pipeline: upload → extract → clean → redact/classify/extract/route → confidence gate. 3. Live upload of the frozen synthetic example and explanation of `needs_review`. 4. Measured scores, limitations, and next step: insurer-text validation plus OCR hardening before any production use.

The M1 three-minute manual-processing assumption is still unsourced and is not used as a verified business saving.
