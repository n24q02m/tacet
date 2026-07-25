"""Render the paper's abstract to the HTML the Zenodo deposit description wants.

The deposit description and the paper's abstract say the same thing, so keeping
two hand-written copies guarantees they drift. This extracts the abstract from a
LaTeXML rendering of ``paper/main.tex`` -- the same source the deposited PDF is
built from.

The whole document is rendered rather than a stub carrying just the abstract:
the abstract cites (``\\citep``), cross-references (``\\ref``) and uses macros
defined in the preamble (``\\akAcc``, ``\\costFactor``, ``\\seeds``), none of
which resolve outside the real document.

Usage::

    python scripts/zenodo_description.py --tex paper/main.tex --out description.html
    python scripts/zenodo_description.py --html rendered.html --out description.html
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def render(tex: Path) -> str:
    """Run LaTeXML over the real document and return the HTML."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "paper.html"
        proc = subprocess.run(
            ["latexmlc", tex.name, "--format=html5", "--nocomments", f"--dest={out}"],
            cwd=tex.parent,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not out.is_file():
            tail = (proc.stderr or proc.stdout or "")[-1500:]
            raise SystemExit(f"latexmlc failed ({proc.returncode}):\n{tail}")
        return out.read_text(encoding="utf-8", errors="replace")


def extract(html: str) -> str:
    """Pull the abstract out of the rendering and reduce it to Zenodo's HTML subset."""
    block = re.search(
        r'<div class="ltx_abstract"[^>]*>(.*?)</div>\s*(?:<div|<section|</article)',
        html,
        re.DOTALL,
    )
    if not block:
        raise SystemExit("no ltx_abstract block in the LaTeXML output")

    paragraphs = re.findall(r"<p\b[^>]*>(.*?)</p>", block.group(1), re.DOTALL)
    if not paragraphs:
        raise SystemExit("the abstract block contained no paragraphs")

    cleaned: list[str] = []
    for para in paragraphs:
        text = para
        # Anchors point at a document that does not exist on the deposit page.
        text = re.sub(r"<a\b[^>]*>(.*?)</a>", r"\1", text, flags=re.DOTALL)
        # LaTeXML marks up maths and inline styling with spans carrying classes;
        # Zenodo accepts a small subset, so keep the text and drop the wrappers.
        text = re.sub(r"</?span\b[^>]*>", "", text)
        text = re.sub(r"</?math\b[^>]*>", "", text)
        text = re.sub(r'\s+class="[^"]*"', "", text)
        text = re.sub(r'\s+id="[^"]*"', "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            cleaned.append(f"<p>{text}</p>")
    return "\n".join(cleaned)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tex", type=Path, help="LaTeX source to render")
    src.add_argument("--html", type=Path, help="an existing LaTeXML rendering")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    html = (
        args.html.read_text(encoding="utf-8", errors="replace") if args.html else render(args.tex)
    )
    description = extract(html)

    # The abstract runs well over a thousand characters; anything near-empty means
    # the extraction matched the wrong block rather than that the paper changed.
    if len(description) < 400:
        print(f"description is implausibly short ({len(description)} chars):", file=sys.stderr)
        print(description[:200], file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(description, encoding="utf-8")
    print(f"wrote {args.out} ({len(description)} chars, {description.count('<p>')} paragraphs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
