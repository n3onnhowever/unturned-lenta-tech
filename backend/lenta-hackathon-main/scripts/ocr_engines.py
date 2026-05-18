"""
Ленивая загрузка нескольких OCR-движков для кропов ценников (BGR uint8 OpenCV).

Поддерживаются (по мере установленных пакетов):
  - paddle   — PaddleOCR (русский rec+det на кропе)
  - easyocr  — EasyOCR ['ru','en']
  - tesseract — pytesseract (нужен бинарник tesseract в PATH, rus+eng)
  - rapidocr — RapidOCR ONNX (rapidocr-onnxruntime)
  - doctr    — docTR DBNet + Parseq (python-doctr[torch])

Единый контракт: run_engine(name, image_bgr) -> OCREngineResult
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class OCRLine:
    text: str
    score: float | None = None


@dataclass
class OCREngineResult:
    engine: str
    lines: list[OCRLine]
    elapsed_sec: float
    error: str | None = None

    @property
    def full_text(self) -> str:
        return "\n".join(l.text.strip() for l in self.lines if l.text and l.text.strip())

    def to_json_lines(self) -> str:
        return json.dumps([asdict(l) for l in self.lines], ensure_ascii=False)


def _bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def ensure_min_ocr_size(img_bgr: np.ndarray, min_side: int = 384) -> np.ndarray:
    """Paddle det often returns 0 boxes on tiny price-tag crops."""
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    side = min(h, w)
    if side >= min_side:
        return img_bgr
    scale = min_side / max(side, 1)
    return cv2.resize(
        img_bgr,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_LANCZOS4,
    )


_CACHED: dict[str, Any] = {}


def tesseract_executable() -> Path | None:
    """Путь к tesseract.exe: TESSERACT_CMD, PATH, или стандартный путь установки Windows."""
    env = (os.environ.get("TESSERACT_CMD") or "").strip().strip('"')
    if env:
        p = Path(env)
        if p.is_file():
            return p
        return None
    w = shutil.which("tesseract")
    if w:
        return Path(w)
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    cand = Path(pf) / "Tesseract-OCR" / "tesseract.exe"
    if cand.is_file():
        return cand
    return None


def _tessdata_extra_dir() -> Path | None:
    """Доп. tessdata (rus+eng): TESSDATA_EXTRA_DIR или ./tessdata_extra в корне репозитория."""
    raw = (os.environ.get("TESSDATA_EXTRA_DIR") or "").strip().strip('"')
    if raw:
        p = Path(raw)
        if p.is_dir() and (p / "rus.traineddata").is_file():
            return p
    here = Path(__file__).resolve().parents[1] / "tessdata_extra"
    if here.is_dir() and (here / "rus.traineddata").is_file():
        return here
    return None


def tesseract_available() -> bool:
    return tesseract_executable() is not None


def detect_installed_engines() -> list[str]:
    """Движки, для которых есть зависимости в текущем окружении (без загрузки весов)."""
    out: list[str] = []

    try:
        import paddleocr  # noqa: F401

        out.append("paddle")
    except ImportError:
        pass

    try:
        import easyocr  # noqa: F401

        out.append("easyocr")
    except ImportError:
        pass

    try:
        import pytesseract  # noqa: F401

        if tesseract_executable() is not None:
            out.append("tesseract")
    except ImportError:
        pass

    try:
        import rapidocr_onnxruntime  # noqa: F401

        out.append("rapidocr")
    except ImportError:
        pass

    try:
        import doctr  # noqa: F401

        out.append("doctr")
    except ImportError:
        pass

    return out


def _get_paddle():
    if "paddle" not in _CACHED:
        from paddleocr import PaddleOCR

        # PaddleOCR 3.x: use_angle_cls устарел; передача ломает внутренний predict(cls=...)
        _CACHED["paddle"] = PaddleOCR(lang="ru")
    return _CACHED["paddle"]


def _get_easyocr():
    if "easyocr" not in _CACHED:
        import easyocr

        _CACHED["easyocr"] = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)
    return _CACHED["easyocr"]


def _get_rapidocr():
    if "rapidocr" not in _CACHED:
        from rapidocr_onnxruntime import RapidOCR

        _CACHED["rapidocr"] = RapidOCR()
    return _CACHED["rapidocr"]


def _get_doctr_predictor():
    if "doctr" not in _CACHED:
        from doctr.models import ocr_predictor

        _CACHED["doctr"] = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="parseq",
            pretrained=True,
            assume_straight_pages=True,
        )
    return _CACHED["doctr"]


def run_paddle(img_bgr: np.ndarray) -> list[OCRLine]:
    ocr = _get_paddle()
    img_bgr = ensure_min_ocr_size(img_bgr)
    rgb = _bgr_to_rgb(img_bgr)
    lines_out: list[OCRLine] = []

    # PaddleOCR 3.x (PaddleX): predict() -> list[OCRResult] с rec_texts / rec_scores
    try:
        pages = ocr.predict(rgb)
    except AttributeError:
        pages = ocr.ocr(rgb)

    if not pages:
        return lines_out

    for page in pages:
        if page is None:
            continue
        if isinstance(page, dict) or (
            hasattr(page, "__getitem__") and "rec_texts" in page
        ):
            try:
                texts = page["rec_texts"]
                scores = page["rec_scores"] if "rec_scores" in page else None
            except (KeyError, TypeError):
                continue
            if scores is None:
                for t in texts:
                    if t:
                        lines_out.append(OCRLine(text=str(t), score=None))
            else:
                for i, t in enumerate(texts):
                    if not t:
                        continue
                    try:
                        sc = float(scores[i])
                    except (TypeError, ValueError, IndexError):
                        sc = None
                    lines_out.append(OCRLine(text=str(t), score=sc))
            continue

        # Формат 2.x: список элементов [box, (text, conf)]
        if isinstance(page, list):
            for item in page:
                if item is None:
                    continue
                try:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        txt_conf = item[1]
                        if isinstance(txt_conf, (list, tuple)) and len(txt_conf) >= 2:
                            text, conf = str(txt_conf[0]), float(txt_conf[1])
                        elif isinstance(txt_conf, str):
                            text, conf = txt_conf, None
                        else:
                            continue
                        lines_out.append(OCRLine(text=text, score=conf))
                except (TypeError, ValueError, IndexError):
                    continue

    return lines_out


def run_easyocr(img_bgr: np.ndarray) -> list[OCRLine]:
    reader = _get_easyocr()
    rgb = _bgr_to_rgb(img_bgr)
    res = reader.readtext(rgb)
    lines_out: list[OCRLine] = []
    for _box, text, conf in res:
        lines_out.append(OCRLine(text=str(text), score=float(conf)))
    return lines_out


def run_tesseract(img_bgr: np.ndarray) -> list[OCRLine]:
    import pytesseract
    from PIL import Image

    exe = tesseract_executable()
    if exe is not None:
        pytesseract.pytesseract.tesseract_cmd = str(exe)

    rgb = _bgr_to_rgb(img_bgr)
    pil = Image.fromarray(rgb)
    extra = _tessdata_extra_dir()
    cfg = ""
    if extra is not None:
        # Без лишних кавычек: иначе Windows передаёт путь с "" и Tesseract не находит .traineddata
        td = extra.resolve().as_posix()
        cfg = f"--tessdata-dir {td}"

    def ocr_with_lang(lang: str) -> list[OCRLine]:
        data = pytesseract.image_to_data(
            pil,
            lang=lang,
            output_type=pytesseract.Output.DICT,
            config=cfg,
        )
        lines_out: list[OCRLine] = []
        n = len(data.get("text", []))
        for i in range(n):
            t = (data["text"][i] or "").strip()
            if not t:
                continue
            try:
                conf = float(data["conf"][i])
            except (ValueError, TypeError):
                conf = None
            if conf is not None and conf < 0:
                conf = None
            lines_out.append(OCRLine(text=t, score=conf))
        return lines_out

    try:
        return ocr_with_lang("rus+eng")
    except Exception:
        return ocr_with_lang("eng")


def run_rapidocr(img_bgr: np.ndarray) -> list[OCRLine]:
    engine = _get_rapidocr()
    rgb = _bgr_to_rgb(img_bgr)
    result, _elapse = engine(rgb)
    lines_out: list[OCRLine] = []
    if not result:
        return lines_out
    for item in result:
        # [box, text, score]
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            text = str(item[1])
            score = float(item[2]) if item[2] is not None else None
            lines_out.append(OCRLine(text=text, score=score))
    return lines_out


def run_doctr(img_bgr: np.ndarray) -> list[OCRLine]:
    from doctr.io import DocumentFile

    predictor = _get_doctr_predictor()
    rgb = _bgr_to_rgb(img_bgr)
    try:
        doc = DocumentFile.from_ndarrays([rgb])
    except Exception:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tpath = Path(tmp.name)
            cv2.imwrite(str(tpath), img_bgr)
        try:
            doc = DocumentFile.from_images([str(tpath)])
        finally:
            tpath.unlink(missing_ok=True)
    out = predictor(doc)
    lines_out: list[OCRLine] = []
    for page in out.pages:
        for block in page.blocks:
            for line in block.lines:
                parts: list[str] = []
                scores: list[float] = []
                for word in line.words:
                    parts.append(word.value)
                    if word.confidence is not None:
                        scores.append(float(word.confidence))
                text = " ".join(parts).strip()
                if not text:
                    continue
                score = sum(scores) / len(scores) if scores else None
                lines_out.append(OCRLine(text=text, score=score))
    return lines_out


def run_engine(engine: str, img_bgr: np.ndarray) -> OCREngineResult:
    engine = engine.strip().lower()
    t0 = time.perf_counter()
    try:
        if engine == "paddle":
            lines = run_paddle(img_bgr)
        elif engine == "easyocr":
            lines = run_easyocr(img_bgr)
        elif engine == "tesseract":
            if not tesseract_available():
                raise RuntimeError(
                    "Нет tesseract: добавьте в PATH или задайте TESSERACT_CMD "
                    "(полный путь к tesseract.exe, см. requirements-ocr.txt)"
                )
            lines = run_tesseract(img_bgr)
        elif engine == "rapidocr":
            lines = run_rapidocr(img_bgr)
        elif engine == "doctr":
            lines = run_doctr(img_bgr)
        else:
            raise ValueError(f"Неизвестный движок: {engine}")
        elapsed = time.perf_counter() - t0
        return OCREngineResult(engine=engine, lines=lines, elapsed_sec=elapsed, error=None)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return OCREngineResult(engine=engine, lines=[], elapsed_sec=elapsed, error=str(e))
