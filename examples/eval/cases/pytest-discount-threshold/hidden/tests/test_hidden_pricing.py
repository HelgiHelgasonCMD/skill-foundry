from pricing import discount


def test_threshold_is_inclusive():
    assert discount(100.0) == 90.0


def test_just_below_threshold():
    assert discount(99.99) == 99.99


def test_large_order():
    assert discount(150.0) == 135.0
