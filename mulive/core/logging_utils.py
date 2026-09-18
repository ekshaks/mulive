import sys
from datetime import datetime, timezone


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def clip_text(text, max_chars=1600):
    text = "" if text is None else str(text)
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n... [clipped {omitted} chars]"


def log_text_block(title, text, max_chars=1600):
    line = "=" * 16
    print(f"\n{line} {title} {line}")
    print(clip_text(text, max_chars=max_chars))
    print(f"{line} END {title} {line}\n")


def monitor_log(message):
    print(f"[monitor] ts={_timestamp()} {message}")


def monitor_time(service, operation, elapsed_s, **fields):
    """Emit one machine-readable latency line for a runtime operation."""
    elapsed = f"elapsed_ms={elapsed_s * 1000:.0f}"
    if sys.stdout.isatty():
        elapsed = f"\033[1m{elapsed}\033[0m"
    details = [
        f"service={service}",
        elapsed,
        f"operation={operation}",
        *(f"{key}={value}" for key, value in fields.items() if value is not None),
    ]
    print(f"[monitor-time] ts={_timestamp()} {' '.join(details)}")
