from openpyxl import load_workbook

from beer_sentiment.io.excel import write_colored_excel
from beer_sentiment.models import Label


def test_excel_style_and_fill(tmp_path, config):
    path = tmp_path / "out.xlsx"
    rows = [{"标题": "a", "正文": "b"}, {"标题": "c", "正文": "d"}]
    write_colored_excel(path, rows, [Label.BLUE, Label.YELLOW], ["标题", "正文"], config)
    workbook = load_workbook(path)
    worksheet = workbook.active
    assert worksheet.title == "舆情"
    assert worksheet["A2"].fill.start_color.rgb == "FF00B0F0"
    assert worksheet["A3"].fill.start_color.rgb == "FFFFFF00"
    assert worksheet["A1"].font.name == "宋体"
    assert worksheet.freeze_panes == "A2"
