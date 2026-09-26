from beer_sentiment.rules.normalize import (
    clean_text,
    extract_brands,
    normalize_ocr_noise,
)


def test_clean_text_removes_bom_and_zero_width():
    assert clean_text("\ufeff百威\u200b ") == "百威"


def test_normalize_ocr_noise():
    assert normalize_ocr_noise("狗兑和勾对都是问题") == "勾兑和勾兑都是问题"


def test_extract_brands_with_alias(config):
    assert extract_brands("百威英博啤酒", config) == ["百威"]
    assert extract_brands("锐澳果啤", config)[0] == "RIO"
    assert extract_brands("雪花啤酒挺好喝", config) == ["雪花"]


def test_extract_brands_with_fuzzy_and_pinyin(config):
    assert "百威" in extract_brands("百微太难喝了", config)
    assert "百威" in extract_brands("baiwei beer", config)


def test_fuzzy_brand_at_other_position_after_exact_brand(config):
    brands = extract_brands("百威和青道啤酒都被投诉", config)
    assert "百威" in brands
    assert "青岛" in brands


def test_product_alias_matches_brand(config):
    assert "雪花" in extract_brands("勇闯天涯卖不动", config)
