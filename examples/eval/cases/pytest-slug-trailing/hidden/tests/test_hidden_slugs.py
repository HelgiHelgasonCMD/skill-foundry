from slugs import slugify


def test_trailing_punctuation():
    assert slugify("Hello, World!") == "hello-world"


def test_leading_punctuation():
    assert slugify("  --Already Slugged-- ") == "already-slugged"


def test_digits_kept():
    assert slugify("Release 2.0") == "release-2-0"
