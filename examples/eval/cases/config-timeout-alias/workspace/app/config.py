"""Settings loader. v2 renamed REQUEST_TIMEOUT to HTTP_TIMEOUT_S."""

DEFAULTS = {"http_timeout_s": 30}


def load(env: dict[str, str]) -> dict[str, object]:
    cfg: dict[str, object] = dict(DEFAULTS)
    if "HTTP_TIMEOUT_S" in env:
        cfg["http_timeout_s"] = env["HTTP_TIMEOUT_S"]
    return cfg
