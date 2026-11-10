"""Semantic shortlisting to limit the number of detailed assessments."""
from __future__ import annotations

from .models import Paper
from .ollama import embed_many
from .references import choose_references
from .text import cosine


def rank_candidates(papers: list[Paper], profile: str, ref_chunks: list[tuple[str, str, list[float]]],
                    embed_model: str, cap: int) -> list[tuple[float, Paper, list[str]]]:
    """Select papers by similarity to the profile (65%) and references (35%).

    Return (similarity, paper, excerpts) tuples up to cap. This selection
    precedes detailed model assessment and does not represent the final score.
    """
    profile_vec = embed_many([profile[:5000]], embed_model)[0]
    selected: list[tuple[float, Paper, list[str]]] = []
    # Reuse one profile vector across all candidate batches.
    for offset in range(0, len(papers), 16):
        page = papers[offset:offset + 16]
        vecs = embed_many([p.title + ". " + p.abstract for p in page], embed_model)
        for p, v in zip(page, vecs):
            profile_sim = cosine(v, profile_vec)
            # Without references, the score reduces to similarity with the profile.
            top_reference = max((cosine(v, row[2]) for row in ref_chunks), default=profile_sim)
            combined = 0.65 * profile_sim + 0.35 * top_reference
            selected.append((combined, p, choose_references(v, ref_chunks)))
    selected.sort(key=lambda item: item[0], reverse=True)
    if len(selected) > cap:
        print(f"Embedding shortlist: selecting {cap}/{len(selected)} for detailed LLM analysis; "
              "increase MAX_LLM_EVALUATIONS if needed.", flush=True)
    return selected[:cap]
