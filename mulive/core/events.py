from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TranscriptEvent:
    text: str
    is_final: bool
    utterance_id: str | None = None
    context: Any = None


@dataclass(frozen=True)
class ClientTranscriptMessage:
    role: str
    content: str

    def to_client_dict(self) -> dict[str, Any]:
        return {
            "type": "transcript",
            "role": self.role,
            "content": self.content,
        }
