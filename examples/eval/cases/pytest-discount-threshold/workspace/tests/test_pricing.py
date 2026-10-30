from pricing import discount


def test_small_orders_pay_full_price():
    assert discount(20.0) == 20.0


def test_discount_threshold():
    assert discount(100.0) == 90.0
