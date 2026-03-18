from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree

try:
    import fitz  # PyMuPDF
except ModuleNotFoundError:
    fitz = None


DOCX_NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def extract_text_from_path(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path), "pdf"
    if suffix == ".docx":
        return extract_text_from_docx(path), "docx"
    if suffix in {".html", ".htm"}:
        return strip_html(path.read_text(errors="replace")), "html_file"
    return path.read_text(errors="replace"), "text_file"


def extract_text_from_pdf(path: Path) -> str:
    if fitz is None:
        return ""
    with fitz.open(path) as document:
        pages = [page.get_text("text") for page in document]
    return normalize_whitespace("\n".join(pages))


def extract_text_from_docx(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            xml_bytes = archive.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile, FileNotFoundError):
        return ""

    root = ElementTree.fromstring(xml_bytes)
    texts = [node.text or "" for node in root.findall(".//w:t", DOCX_NAMESPACE)]
    return normalize_whitespace(" ".join(texts))


def strip_html(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return normalize_whitespace(without_tags)


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
