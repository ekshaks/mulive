"""Output packet every app controller emits.

Apps emit one ``AppOutput`` per controller step. The session pipeline fans it out
to three places: ``messages`` go to the transcript and TTS, ``feedback`` goes to
the browser, and ``effect`` is picked up by an :mod:`mulive.apps.effects` runner.

``spell`` still carries its own ``GameOutput`` copy; new apps should import from
here.
"""

from dataclasses import dataclass, field
from typing import Any

from mulive.apps.events import FeedbackEvent


@dataclass
class AppOutput:
    """One packet of controller output.

    Attributes:
        messages: Lines to show in the transcript and speak. Empties are dropped.
        state: The controller's state object, passed through untouched.
        finished: True when the session's activity is over.
        effect: A request record for an async side effect (engine, LLM, OCR).
        feedback: A structured message for the browser UI.
    """

    messages: list[str] = field(default_factory=list)
    state: Any = None
    finished: bool = False
    effect: Any = None
    feedback: FeedbackEvent | None = None


def output(
    state: Any = None,
    *messages: str,
    finished: bool = False,
    effect: Any = None,
    feedback: FeedbackEvent | None = None,
) -> AppOutput:
    """Build an ``AppOutput``, dropping empty messages.

    Args:
        state: Controller state to pass through.
        *messages: Spoken/displayed lines; falsy entries are skipped.
        finished: Marks the activity as over.
        effect: Async effect request to hand to an effect runner.
        feedback: Structured UI message.

    Returns:
        The assembled ``AppOutput``.
    """
    return AppOutput(
        messages=[message for message in messages if message],
        state=state,
        finished=finished,
        effect=effect,
        feedback=feedback,
    )
