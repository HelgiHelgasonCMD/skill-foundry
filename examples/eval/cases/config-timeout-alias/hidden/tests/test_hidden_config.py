from app.config import load


def test_new_name_coerced():
    assert load({"HTTP_TIMEOUT_S": "7"})["http_timeout_s"] == 7


def test_deprecated_alias_still_works():
    assert load({"REQUEST_TIMEOUT": "5"})["http_timeout_s"] == 5


def test_new_name_wins_over_alias():
    assert load({"REQUEST_TIMEOUT": "5", "HTTP_TIMEOUT_S": "9"})["http_timeout_s"] == 9


def test_default():
    assert load({})["http_timeout_s"] == 30
