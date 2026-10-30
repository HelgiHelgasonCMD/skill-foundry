from app.config import load


def test_default_timeout():
    assert load({})["http_timeout_s"] == 30


def test_timeout_from_env_is_int():
    assert load({"HTTP_TIMEOUT_S": "7"})["http_timeout_s"] == 7
