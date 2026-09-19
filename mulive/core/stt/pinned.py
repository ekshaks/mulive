"""One dedicated worker thread for a local Whisper model."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..logging_utils import monitor_log
from .config import STTConfig


class PinnedWhisper:
    """Load and run a local Whisper model on a single dedicated thread.

    Two reasons for a pinned thread rather than :func:`asyncio.to_thread`:

    * MLX expects its model to be created and used on the same thread.
    * One worker means one transcription at a time. A timeout cannot actually stop
      Whisper once it has started, so without that bound an abandoned call would
      keep a core busy while the next segment starts a second one — on a small box
      that is how one slow utterance turns into a permanent backlog.

    Loading starts when :meth:`wait_ready` or the first transcription is requested.
    Server startup calls :meth:`wait_ready`, while merely constructing an embedded
    runtime stays side-effect free.

    Args:
        config: Local Whisper provider and model configuration.
    """

    def __init__(self, config: STTConfig):
        self.config = config
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"{config.provider}-whisper"
        )
        self._stt_future = None

    def _start_loading(self):
        if self._stt_future is None:
            self._stt_future = self.executor.submit(self._load)
            self._stt_future.add_done_callback(self._log_load)
        return self._stt_future

    def _load(self):
        """Build the model. Runs on the worker thread."""
        from .whisper import WhisperSTT

        return WhisperSTT(self.config)

    def _log_load(self, future) -> None:
        """Log a failed load as soon as it happens, not on the first utterance."""
        if future.cancelled():
            return
        error = future.exception()
        if error is not None:
            monitor_log(
                f"stt model load failed provider={self.config.provider} model={self.config.model_label} "
                f"error={type(error).__name__}: {error}"
            )

    def _infer(self, segment):
        """Transcribe one segment. Runs on the worker thread."""
        return self._stt_future.result().transcribe_turn(segment)

    def is_loading(self) -> bool:
        """True while the model is still being loaded."""
        return self._stt_future is not None and not self._stt_future.done()

    async def transcribe_turn(self, segment):
        """Transcribe one audio segment on the worker thread.

        Args:
            segment: 16 kHz mono int16 samples.

        Returns:
            The recognised text.
        """
        self._start_loading()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._infer, segment)

    async def wait_ready(self) -> None:
        """Wait until the model has loaded, propagating load failures."""
        await asyncio.wrap_future(self._start_loading())

    def shutdown(self) -> None:
        """Release the worker thread and drop any queued work."""
        self.executor.shutdown(wait=False, cancel_futures=True)
