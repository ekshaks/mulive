"""Events owned by the application-facing server boundary."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FeedbackEvent:
    """Structured application feedback delivered to a client UI."""

    name: str
    result: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_client_dict(self) -> dict[str, Any]:
        return {
            "type": "app_feedback",
            "name": self.name,
            "result": self.result,
            "data": self.data,
        }
