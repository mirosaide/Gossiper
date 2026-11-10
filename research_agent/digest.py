"""Plain-text and HTML reports, independent of network access and email delivery."""
from __future__ import annotations

import html

from .models import Evaluated


def render_digest(items: list[Evaluated], date: str, scanned: int, threshold: int) -> tuple[str, str]:
    """Build plain-text and HTML reports, including the complete paper abstracts."""
    lines = [f"arXiv research digest - {date}",
             f"Searched {scanned} new candidate paper(s); {len(items)} scored >= {threshold}/100.",
             "Assessment uses the arXiv abstracts, NOT independent verification of the full papers.", ""]
    cards = []
    for rank, item in enumerate(items, 1):
        p, a = item.paper, item.assessment
        lines.extend([
            f"{rank}. {p.title} (relevance {a['score']}/100; confidence {a['confidence']})",
            f"Authors: {p.authors}", f"Published: {p.published} | {p.categories}",
            f"Paper: {p.url}", f"Why relevant: {a['why_relevant']}",
            f"Direct comparison: {a['direct_comparison']}",
            f"Research implication: {a['research_implication']}",
            f"Caveats: {a['limitations']}", f"Abstract: {p.abstract}", "", "---", "",
        ])
        # Escape metadata and model responses before inserting them into HTML.
        esc = lambda value: html.escape(str(value), quote=True)
        cards.append(
            '<section class="paper">'
            f'<div class="score">{rank:02d} &middot; RELEVANCE {a["score"]}/100 &middot; '
            f'CONFIDENCE {esc(a["confidence"])}</div>'
            f'<h2><a href="{esc(p.url)}">{esc(p.title)}</a></h2>'
            f'<p class="meta">{esc(p.authors)} &middot; {esc(p.published)} &middot; {esc(p.categories)}</p>'
            f'<p><strong>Why relevant</strong> {esc(a["why_relevant"])}</p>'
            f'<p><strong>Direct comparison with your research</strong> {esc(a["direct_comparison"])}</p>'
            f'<p><strong>Possible next step</strong> {esc(a["research_implication"])}</p>'
            f'<p class="caveat"><strong>Uncertainties</strong> {esc(a["limitations"])}</p>'
            f'<div class="abstract"><strong>Full arXiv abstract</strong><p>{esc(p.abstract)}</p></div>'
            f'<p><a href="{esc(p.url)}">Read paper on arXiv</a></p>'
            '</section>'
        )
    if not items:
        lines.append("No new papers met the configured relevance threshold today.")
        cards.append("<p>No new papers met the configured relevance threshold today.</p>")
    # Inline CSS keeps the preview self-contained without external resources.
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{{background:#f5f6f8;color:#1d2834;font:15px/1.65 system-ui,-apple-system,Segoe UI,Arial,sans-serif;margin:0;padding:30px 12px}}
main{{max-width:760px;margin:auto;background:white;padding:24px 32px;border-radius:12px}}
h1{{font-size:26px;line-height:1.25}}h2{{font-size:19px;line-height:1.35;margin:10px 0}}
a{{color:#1559a0;text-decoration:none}}.meta{{color:#6c7784;font-size:13px}}
.paper{{border-top:1px solid #ddd;padding:23px 0}}.score{{font-size:12px;letter-spacing:.05em;color:#346b64;font-weight:700}}
.caveat{{color:#536170}}.abstract{{background:#f7f9fb;padding:10px 14px;border-radius:6px}}
.footer{{border-top:1px solid #ddd;margin-top:20px;padding-top:14px;color:#77828b;font-size:12px}}
</style></head><body><main>
<div style="text-transform:uppercase;letter-spacing:0.12em;color:#5f6a76;font-weight:700;font-size:12px">Personal research radar</div>
<h1>arXiv research digest</h1>
<p>{html.escape(date)} &middot; {scanned} new candidate paper(s) &middot; {len(items)} above {threshold}/100</p>
{''.join(cards)}
<div class="footer">Automatically generated using arXiv metadata and a local AI model. Evaluations are abstract-based and may be wrong; verify methods and results in the original papers.</div>
</main></body></html>"""
    return "\n".join(lines), page
