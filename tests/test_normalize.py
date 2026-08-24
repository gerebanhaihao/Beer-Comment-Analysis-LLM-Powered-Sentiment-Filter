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
