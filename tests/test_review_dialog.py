"""Windows-only check of the manual review window without a model API."""

import sys
import tkinter as tk

import pytest

from beer_sentiment.gui import ReviewDialog
from beer_sentiment.models import JudgedRow, JudgeResult, Label, PreparedRow, Stage1Result


@pytest.mark.skipif(sys.platform != "win32", reason="桌面窗口只在 Windows 上检查")
def test_review_dialog_requires_a_decision_for_every_uncertain_row():
    root = tk.Tk()
    root.withdraw()
    decisions = []
    rows = [
        JudgedRow(
            prepared=PreparedRow({}, "source.csv", line, "待判断的啤酒评论", Stage1Result(True)),
            result=JudgeResult(label, 0.2, "模型把握不足"),
            low_confidence=True,
        )
        for line, label in ((2, Label.YELLOW), (3, Label.NONE))
    ]
    try:
        dialog = ReviewDialog(root, "source.csv", rows, decisions.append)
        root.update()
        assert dialog.confirm_button.instate(["disabled"])

        dialog.table.selection_set("2")
        dialog.choice.set(Label.BLUE.value)
        dialog._set_decision()
        assert dialog.confirm_button.instate(["disabled"])

        dialog.table.selection_set("3")
        dialog.choice.set(Label.NONE.value)
        dialog._set_decision()
        assert dialog.confirm_button.instate(["!disabled"])
        dialog._confirm()
        assert decisions == [{2: Label.BLUE, 3: Label.NONE}]
    finally:
        root.destroy()
