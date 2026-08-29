"""Stage 1: time-window filtering and rule-based candidate selection."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from beer_sentiment.config import AppConfig
from beer_sentiment.io.csv_io import (
    infer_file_output_class,
    parse_time,
)
from beer_sentiment.models import PreparedRow
from beer_sentiment.rules.classify import Stage1Classifier
from beer_sentiment.rules.normalize import clean_text


@dataclass
class SourcePreparation:
    """Everything needed to run Stage 2 and write the Excel output."""

    source_file: str
    output_class: str
    prepared_rows: list[PreparedRow]


class Stage1Pipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.classifier = Stage1Classifier(config)

    def prepare(
        self,
        rows: list[dict[str, Any]],
        time_col: str,
        data_col: str,
        text_cols: list[str],
        source_file: str,
        start: dt.datetime | None,
        end: dt.datetime | None,
    ) -> SourcePreparation:
        output_class = infer_file_output_class(rows, data_col, source_file)
        prepared_rows: list[PreparedRow] = []
        for original_row_number, row in enumerate(rows, start=2):
            if start is not None and end is not None:
                parsed = parse_time(row.get(time_col, ""))
                if parsed is None or not (start <= parsed <= end):
                    continue
            texts = [clean_text(row.get(header, "")) for header in text_cols]
            combined = "\n".join(text for text in texts if text)
            stage1 = self.classifier.classify(combined)
            prepared_rows.append(
                PreparedRow(
                    row=row,
                    source_file=source_file,
                    original_row_number=original_row_number,
                    combined_text=combined,
                    stage1=stage1,
                )
            )
        return SourcePreparation(
            source_file=source_file,
            output_class=output_class,
            prepared_rows=prepared_rows,
        )
