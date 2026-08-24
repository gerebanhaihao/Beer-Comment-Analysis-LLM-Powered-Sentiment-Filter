from beer_sentiment.io.filenames import output_filename, review_filename


def test_output_filename():
    assert (
        output_filename("quark__2026-08-11 103809.csv", "行业", "morning")
        == "行业上午__2026-08-11 103809.xlsx"
    )
    assert (
        output_filename("quark__2026-08-11 103809.csv", "品牌", "afternoon")
        == "品牌下午__2026-08-11 103809.xlsx"
    )
    assert output_filename("random.csv", "品牌", "afternoon") == "品牌下午.xlsx"


def test_review_filename():
    assert review_filename("morning") == "待筛选_上午.csv"
    assert review_filename("afternoon") == "待筛选_下午.csv"
