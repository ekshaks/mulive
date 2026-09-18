"""One dedicated worker thread for a local Whisper model."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..logging_utils import monitor_log


class PinnedWhisper:
    """Load and run a local Whisper model on a single dedicated thread.

    Two reasons for a pinned thread rather than :func:`asyncio.to_thread`:

    * MLX expects its model to be created and used on the same thread.
    * One worker means one transcription at a time. A timeout cannot actually stop
      Whisper once it has started, so without that bound an abandoned call would
      keep a core busy while the next segment starts a second one — on a small box
      that is how one slow utterance turns into a permanent backlog.

    Loading starts immediately, on the worker, so the event loop never blocks on it
    and :meth:`is_loading` can tell the browser to wait.

    Args:
        mode: ``mlx`` or ``faster_whisper``.
        model_size: Whisper model size or id.
        **kwargs: Backend options passed to :class:`~mulive.core.stt.whisper.WhisperSTT`.
    """

    def __init__(self, mode: str = "mlx", model_size: str = "tiny", **kwargs):
        self.mode = mode
        self.model_size = model_size
        self.kwargs = kwargs
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{mode}-whisper")
        self._stt_future = self.executor.submit(self._load)
        self._stt_future.add_done_callback(self._log_load)

    def _load(self):
        """Build the model. Runs on the worker thread."""
        from .whisper import WhisperSTT

        return WhisperSTT(mode=self.mode, model_size=self.model_size, **self.kwargs)

    def _log_load(self, future) -> None:
        """Log a failed load as soon as it happens, not on the first utterance."""
        if future.cancelled():
            return
        error = future.exception()
        if error is not None:
            monitor_log(
                f"stt model load failed mode={self.mode} model={self.model_size} "
                f"error={type(error).__name__}: {error}"
            )

    def _infer(self, segment):
        """Transcribe one segment. Runs on the worker thread."""
        return self._stt_future.result().transcribe_turn(segment)

    def is_loading(self) -> bool:
        """True while the model is still being loaded."""
        return not self._stt_future.done()

    async def transcribe_turn(self, segment):
        """Transcribe one audio segment on the worker thread.

        Args:
            segment: 16 kHz mono int16 samples.

        Returns:
            The recognised text.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._infer, segment)

    async def wait_ready(self) -> None:
        """Wait until the model has loaded, propagating load failures."""
        await asyncio.wrap_future(self._stt_future)

    def shutdown(self) -> None:
        """Release the worker thread and drop any queued work."""
        self.executor.shutdown(wait=False, cancel_futures=True)
