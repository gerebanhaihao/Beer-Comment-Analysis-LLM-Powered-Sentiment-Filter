from beer_sentiment.models import Label
from beer_sentiment.rules.classify import Stage1Classifier


def test_candidate_filter(config):
    classifier = Stage1Classifier(config)
    assert classifier.classify("今天天气不错").is_candidate is False
    result = classifier.classify("现在的啤酒都是勾兑的吗")
    assert result.is_candidate is True
    assert "勾兑" in result.matched_keywords


def test_hint_blue_for_own_brand(config):
    classifier = Stage1Classifier(config)
    result = classifier.classify("百威太难喝了")
    assert result.hint_label == Label.BLUE


def test_hint_yellow_for_competitor(config):
    classifier = Stage1Classifier(config)
    result = classifier.classify("青岛抽检不合格")
    assert result.hint_label == Label.YELLOW


def test_hint_yellow_for_industry_mentioning_own(config):
    classifier = Stage1Classifier(config)
    result = classifier.classify("行业销量下滑，百威增速放缓")
    assert result.hint_label == Label.YELLOW
