from __future__ import annotations

from typing import List

from system.models import PipelineState, ThemeRank


def apply(state: PipelineState) -> None:
    ranked = sorted(state.scores, key=lambda item: item.score, reverse=True)
    ranks: List[ThemeRank] = []
    for idx, item in enumerate(ranked, start=1):
        ranks.append(ThemeRank(theme=item.theme, rank=idx, score=item.score))
    state.ranks = ranks
