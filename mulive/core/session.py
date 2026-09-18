import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict

from .audio_input import AudioInput
from .audio_output import AudioOutput


@dataclass
class SessionContext:
    """Runtime resources and lifecycle signals for one live session."""

    pc: Any
    audio_output: AudioOutput
    data_channels: Dict[str, Any]
    audio_input: AudioInput
    video_input: Any
    client_input: Any
    main_loop: asyncio.AbstractEventLoop
    user_id: str | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    closed: asyncio.Event = field(default_factory=asyncio.Event)

    async def wait_until_ready(self) -> None:
        await self.ready.wait()

    def send_to_client(self, message: Any, channel: str = "server_text") -> bool:
        """Serialize a message and send it on a client data channel.

        ``message`` is either a plain JSON-serializable value or an object
        with ``to_client_dict()``. Returns ``False`` when the channel is
        missing or not open. Normally every caller already runs on the main
        event loop; a thread-safe hop is kept as insurance for stray callers.
        """
        data_channel = self.data_channels.get(channel)
        if data_channel is None or data_channel.readyState != "open":
            return False
        payload = message.to_client_dict() if hasattr(message, "to_client_dict") else message
        data = json.dumps(payload)
        try:
            on_loop = asyncio.get_running_loop() is self.main_loop
        except RuntimeError:
            on_loop = False
        if on_loop:
            data_channel.send(data)
        else:
            self.main_loop.call_soon_threadsafe(data_channel.send, data)
        return True

    @property
    def assistant_audio_track(self):
        """Compatibility alias for callers not yet using ``audio_output``."""
        return self.audio_output
