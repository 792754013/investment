"""主题排序：按机会分从高到低生成排名。"""

from __future__ import annotations

from typing import List

from system.models import PipelineState, ThemeRank


def apply(state: PipelineState) -> None:
    # 对评分结果降序排序，分数高的排在前面
    ranked = sorted(state.scores, key=lambda item: item.score, reverse=True)
    ranks: List[ThemeRank] = []
    for idx, item in enumerate(ranked, start=1):
        # 排名从 1 开始，符合常见业务报表习惯
        ranks.append(ThemeRank(theme=item.theme, rank=idx, score=item.score))
    # 保存排名结果
    state.ranks = ranks
