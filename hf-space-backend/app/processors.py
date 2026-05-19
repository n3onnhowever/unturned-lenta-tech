from __future__ import annotations

from typing import Any

from .ml_bundle import process_ml_classify, process_ml_detect, process_ml_finalize, process_ml_ocr


PROCESSORS = {
    "detect": process_ml_detect,
    "classify": process_ml_classify,
    "ocr": process_ml_ocr,
    "finalize": process_ml_finalize,
}
