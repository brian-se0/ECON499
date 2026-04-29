from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
PDF_DIR = ROOT / "lit_review"
OUT_DIR = ROOT / "research" / "consults" / "gpt55-pro-audit"
DOSSIER_PATH = OUT_DIR / "literature-dossier.md"
GAPS_PATH = OUT_DIR / "literature-gaps.md"

MAX_EXTRACT_PAGES = 4
MAX_EXCERPT_CHARS = 1_600


@dataclass(frozen=True)
class PaperEntry:
    path: Path
    sha256: str
    size_mb: float
    page_count: int | None
    metadata_title: str | None
    metadata_author: str | None
    years: tuple[str, ...]
    topic_tags: tuple[str, ...]
    abstract_or_excerpt: str
    extraction_error: str | None


TOPIC_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("arbitrage", (r"\barbitrage\b", r"\bno-arbitrage\b", r"\barbitrage-free\b")),
    ("SVI/eSSVI", (r"\bsvi\b", r"\bessvi\b", r"\bssvi\b")),
    (
        "neural/deep",
        (r"\bneural\b", r"\bdeep learning\b", r"\bdeep neural\b", r"\bnetwork\b", r"\bgan\b"),
    ),
    ("forecasting", (r"\bforecast", r"\bprediction\b", r"\bpredict", r"\bout-of-sample\b")),
    (
        "functional/factor",
        (r"\bfunctional\b", r"\bfactor\b", r"\bprincipal component\b", r"\bpca\b"),
    ),
    ("HAR/time-series", (r"\bhar\b", r"\bheterogeneous autoregressive\b", r"\bautoregressive\b")),
    (
        "evaluation/stat-tests",
        (r"\bmodel confidence set\b", r"\bdiebold\b", r"\bmariano\b", r"\bspa test\b"),
    ),
    (
        "hedging/trading",
        (r"\bhedging\b", r"\btrading strategy\b", r"\boption trading\b", r"\bdelta hedge\b"),
    ),
    (
        "surface construction",
        (r"\bimplied volatility surface\b", r"\bsmoothing\b", r"\binterpolation\b"),
    ),
    ("tail/risk", (r"\btail risk\b", r"\bfomc\b", r"\brisk premium\b", r"\bvolatility risk\b")),
)


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def useful_metadata_value(value: object) -> str | None:
    if value is None:
        return None
    text = normalize_text(str(value))
    if not text:
        return None
    if text.lower() in {"untitled", "unknown"}:
        return None
    return text


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_initial_text(reader: PdfReader) -> str:
    parts: list[str] = []
    for page in reader.pages[: min(MAX_EXTRACT_PAGES, len(reader.pages))]:
        page_text = page.extract_text() or ""
        if page_text.strip():
            parts.append(page_text)
    return normalize_text("\n\n".join(parts))


def extract_years(*values: str | None) -> tuple[str, ...]:
    years: set[str] = set()
    for value in values:
        if value:
            years.update(re.findall(r"\b(?:19|20)\d{2}\b", value))
    return tuple(sorted(years))


def extract_abstract(text: str) -> str:
    lowered = text.lower()
    abstract_at = lowered.find("abstract")
    if abstract_at >= 0:
        tail = text[abstract_at:]
        end_match = re.search(
            r"\n\s*(?:keywords?|jel classification|1\.?\s+introduction|introduction)\b",
            tail,
            flags=re.IGNORECASE,
        )
        if end_match:
            tail = tail[: end_match.start()]
        return normalize_text(tail[:MAX_EXCERPT_CHARS])
    return normalize_text(text[:MAX_EXCERPT_CHARS])


def detect_topics(path: Path, text: str) -> tuple[str, ...]:
    haystack = f"{path.name}\n{text}".lower()
    tags = [
        label
        for label, patterns in TOPIC_PATTERNS
        if any(re.search(pattern, haystack) for pattern in patterns)
    ]
    return tuple(tags) if tags else ("unclassified",)


def read_entry(path: Path) -> PaperEntry:
    sha256 = hash_file(path)
    size_mb = path.stat().st_size / (1024 * 1024)
    try:
        reader = PdfReader(str(path))
        page_count: int | None = len(reader.pages)
        metadata = reader.metadata
        metadata_title = useful_metadata_value(metadata.title if metadata else None)
        metadata_author = useful_metadata_value(metadata.author if metadata else None)
        text = extract_initial_text(reader)
        years = extract_years(path.name, metadata_title, metadata_author, text[:3_000])
        return PaperEntry(
            path=path,
            sha256=sha256,
            size_mb=size_mb,
            page_count=page_count,
            metadata_title=metadata_title,
            metadata_author=metadata_author,
            years=years,
            topic_tags=detect_topics(path, text),
            abstract_or_excerpt=extract_abstract(text),
            extraction_error=None,
        )
    except (OSError, ValueError) as exc:
        return PaperEntry(
            path=path,
            sha256=sha256,
            size_mb=size_mb,
            page_count=None,
            metadata_title=None,
            metadata_author=None,
            years=extract_years(path.name),
            topic_tags=("extraction-error",),
            abstract_or_excerpt="",
            extraction_error=f"{type(exc).__name__}: {exc}",
        )


def markdown_escape(text: str) -> str:
    return text.replace("|", "\\|")


def render_dossier(entries: list[PaperEntry]) -> str:
    tag_counts: dict[str, int] = {}
    for entry in entries:
        for tag in entry.topic_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    lines: list[str] = [
        "# Local Literature Dossier",
        "",
        "Generated: 2026-04-27",
        f"Source directory: `{PDF_DIR}`",
        f"PDF count: {len(entries)}",
        "",
        "This dossier is an extracted local-literature catalog for GPT 5.5 Pro. "
        "It is not a substitute for reading a specific paper when a finding depends on a "
        "precise theorem, data sample, or empirical result.",
        "",
        "## Topic Coverage",
        "",
    ]
    for tag, count in sorted(tag_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {tag}: {count}")
    lines.extend(
        [
            "",
            "## Catalog",
            "",
            "| ID | File | Pages | Years | Tags | Metadata title |",
            "| --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for index, entry in enumerate(entries, start=1):
        title = markdown_escape(entry.metadata_title or "")
        years = ", ".join(entry.years)
        tags = ", ".join(entry.topic_tags)
        lines.append(
            f"| LIT-{index:03d} | `{entry.path.relative_to(ROOT)}` | "
            f"{entry.page_count if entry.page_count is not None else 'n/a'} | "
            f"{markdown_escape(years)} | {markdown_escape(tags)} | {title} |"
        )

    lines.extend(["", "## Extracted Entry Notes", ""])
    for index, entry in enumerate(entries, start=1):
        rel_path = entry.path.relative_to(ROOT)
        title = entry.metadata_title or rel_path.stem
        lines.extend(
            [
                f"### LIT-{index:03d}: {title}",
                "",
                f"- File: `{rel_path}`",
                f"- SHA256: `{entry.sha256}`",
                f"- Size: {entry.size_mb:.2f} MB",
                f"- Pages: {entry.page_count if entry.page_count is not None else 'n/a'}",
                f"- Years found: {', '.join(entry.years) if entry.years else 'none detected'}",
                f"- Topic tags: {', '.join(entry.topic_tags)}",
                f"- Metadata author: {entry.metadata_author or 'none detected'}",
            ]
        )
        if entry.extraction_error:
            lines.append(f"- Extraction error: {entry.extraction_error}")
        lines.extend(
            [
                "",
                "Excerpt:",
                "",
                "```text",
                entry.abstract_or_excerpt or "No extractable text found in the first pages.",
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_gaps(entries: list[PaperEntry]) -> str:
    tag_counts: dict[str, int] = {}
    for entry in entries:
        for tag in entry.topic_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    lines = [
        "# Literature Gaps For Pro Review",
        "",
        "Generated: 2026-04-27",
        "",
        "Initial heuristic only. GPT 5.5 Pro should confirm from the dossier and request "
        "targeted additional literature only for concrete gaps.",
        "",
        "## Candidate Gaps To Check",
        "",
        "- Whether the local set sufficiently covers 15:45 same-day option-data timing "
        "constraints and early-close handling.",
        "- Whether the local set sufficiently supports the exact benchmark suite: no-change "
        "surface, ridge, elastic net, HAR/factor, LightGBM/random forest if retained, and "
        "arbitrage-aware neural total-variance forecasting.",
        "- Whether the local set sufficiently supports observed-cell versus completed-grid "
        "evaluation and mask preservation.",
        "- Whether the local set sufficiently supports the selected statistical comparison "
        "procedures and multiple-model inference.",
        "- Whether the local set sufficiently supports QLIKE-style positivity handling and "
        "total-variance loss choices.",
        "",
        "## Heuristic Topic Counts",
        "",
    ]
    for tag, count in sorted(tag_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {tag}: {count}")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"Missing literature directory: {PDF_DIR}")
    entries = [read_entry(path) for path in sorted(PDF_DIR.glob("*.pdf"))]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOSSIER_PATH.write_text(render_dossier(entries), encoding="utf-8")
    GAPS_PATH.write_text(render_gaps(entries), encoding="utf-8")
    print(f"Wrote {DOSSIER_PATH}")
    print(f"Wrote {GAPS_PATH}")
    print(f"PDFs processed: {len(entries)}")


if __name__ == "__main__":
    main()
