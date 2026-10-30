from slugs import slugify


def test_basic():
    assert slugify("Hello World") == "hello-world"


def test_punctuation_is_trimmed():
    assert slugify("Hello, World!") == "hello-world"
