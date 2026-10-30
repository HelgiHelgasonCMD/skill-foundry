"""Order pricing rules."""


def discount(total: float) -> float:
    """Apply a 10% discount to orders of 100.00 or more."""
    if total > 100:
        return round(total * 0.9, 2)
    return total
