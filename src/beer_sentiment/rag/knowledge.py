"""知识库：沉淀判定规则（rule）与 Few-shot 示例（example）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class KnowledgeEntry:
    """知识库中的一条规则或示例。"""

    id: str
    type: str  # "rule" | "example"
    text: str
    label: str = ""  # 示例的金标：blue / yellow / none
    reason: str = ""
    tags: list[str] = field(default_factory=list)

    def render(self) -> str:
        if self.type == "rule":
            return f"[规则] {self.text}"
        label = self.label or "none"
        reason = f"（理由：{self.reason}）" if self.reason else ""
        return f"[示例|{label}] {self.text} {reason}".strip()


class KnowledgeBase:
    def __init__(self, entries: list[KnowledgeEntry]) -> None:
        self.entries = entries
        self._index = {entry.id: entry for entry in entries}

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, entry_id: str) -> KnowledgeEntry | None:
        return self._index.get(entry_id)

    def format_context(self, entries: list[KnowledgeEntry]) -> str:
        """把检索到的规则与示例渲染成可拼进 Prompt 的参考块。"""
        lines = ["知识库检索到的判定规则与示例（越靠前越相关）："]
        for entry in entries:
            lines.append(entry.render())
        return "\n".join(lines)

    @classmethod
    def from_yaml(cls, path: str | Path) -> KnowledgeBase:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        entries: list[KnowledgeEntry] = []
        for index, item in enumerate(data.get("entries", []), start=1):
            entries.append(
                KnowledgeEntry(
                    id=str(item.get("id") or f"kb-{index}"),
                    type=str(item.get("type") or "example"),
                    text=str(item.get("text") or ""),
                    label=str(item.get("label") or ""),
                    reason=str(item.get("reason") or ""),
                    tags=[str(tag) for tag in (item.get("tags") or [])],
                )
            )
        if not entries:
            raise ValueError(f"知识库为空：{path}")
        return cls(entries)
