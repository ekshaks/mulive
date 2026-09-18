import os
from pathlib import Path


def _default_cert_dir():
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_home / "mulive" / "certs"


def ssl_config(keyfile=None, certfile=None):
    """Resolve TLS files independently of the current Git worktree."""
    cert_dir = _default_cert_dir()
    ssl_keyfile = Path(
        keyfile or os.environ.get("MULIVE_SSL_KEYFILE", cert_dir / "key.pem")
    ).expanduser()
    ssl_certfile = Path(
        certfile or os.environ.get("MULIVE_SSL_CERTFILE", cert_dir / "cert.pem")
    ).expanduser()

    using_defaults = (
        keyfile is None
        and certfile is None
        and "MULIVE_SSL_KEYFILE" not in os.environ
        and "MULIVE_SSL_CERTFILE" not in os.environ
    )
    if using_defaults and not (ssl_keyfile.exists() and ssl_certfile.exists()):
        legacy_dir = Path(__file__).parent.parent.resolve() / "certs"
        legacy_keyfile = legacy_dir / "key.pem"
        legacy_certfile = legacy_dir / "cert.pem"
        if legacy_keyfile.exists() and legacy_certfile.exists():
            ssl_keyfile, ssl_certfile = legacy_keyfile, legacy_certfile

    for label, path in (("key", ssl_keyfile), ("certificate", ssl_certfile)):
        if not path.is_file():
            raise FileNotFoundError(
                f"SSL {label} file not found: {path}. "
                "Set MULIVE_SSL_KEYFILE/MULIVE_SSL_CERTFILE or use --http."
            )
    return {
        "ssl_keyfile": str(ssl_keyfile),
        "ssl_certfile": str(ssl_certfile),
    }


def web_config(use_https: bool = True, **overrides):
    ssl_keyfile = overrides.pop("ssl_keyfile", None)
    ssl_certfile = overrides.pop("ssl_certfile", None)
    config = {
        "debug": False,
    }
    if use_https:
        config.update(ssl_config(ssl_keyfile, ssl_certfile))
    config.update(overrides)
    return config
