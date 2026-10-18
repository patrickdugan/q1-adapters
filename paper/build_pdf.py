"""Render paper/q1_adapters.md to paper/q1_adapters.pdf (Markdown -> styled HTML -> headless Edge print).

  python paper/build_pdf.py        (needs the `markdown` package and Microsoft Edge)
"""

import subprocess
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
SRC = HERE / "q1_adapters.md"
HTML = HERE / "q1_adapters.html"
PDF = HERE / "q1_adapters.pdf"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = """
@page { size: Letter; margin: 0.8in 0.75in; }
body { font-family: 'Cambria', 'Georgia', serif; font-size: 10.5pt; line-height: 1.38; color: #111; max-width: 100%; }
h1 { font-size: 17pt; line-height: 1.2; margin: 0 0 0.4em; text-align: center; }
h1 + p, h1 + p + p { text-align: center; }
h2 { font-size: 13pt; margin: 1.3em 0 0.4em; border-bottom: 1px solid #bbb; padding-bottom: 2px; }
h3 { font-size: 11pt; margin: 1.1em 0 0.3em; }
p { margin: 0.45em 0; text-align: justify; }
code { font-family: 'Consolas', monospace; font-size: 8.8pt; background: #f3f3f3; padding: 0 2px; }
pre { background: #f6f6f6; padding: 6px 8px; font-size: 8.5pt; white-space: pre-wrap; }
table { border-collapse: collapse; margin: 0.6em 0; font-size: 8.4pt; width: 100%; page-break-inside: avoid; }
th, td { border-top: 1px solid #ccc; border-bottom: 1px solid #ccc; padding: 2px 4px; vertical-align: top; text-align: left; }
th { background: #f0f0f0; }
img { max-width: 100%; display: block; margin: 0.6em auto; }
ul, ol { margin: 0.3em 0 0.3em 1.2em; padding-left: 0.6em; }
li { margin: 0.15em 0; }
a { color: #1a4d8f; text-decoration: none; }
"""


def main():
    md = SRC.read_text(encoding="utf-8")
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    HTML.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>q1-adapters</title><style>{CSS}</style>"
                    f"</head><body>{body}</body></html>", encoding="utf-8")
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={PDF}",
                    HTML.as_uri()], check=True, timeout=180)
    print(f"wrote {PDF} ({PDF.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
