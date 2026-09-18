"""One structure for every question an app answers about its own state.

Apps keep asking the same shape of question in many disguises. In chess: "what's
the best move?", "what if I take with the knight?", "why was that bad?", and the
blunder guard's silent "is the move he just chose a mistake?". In spell: "how do
I spell it?", "give me a hint", "why was that wrong?".

They are all: *take a snapshot of the state, optionally apply a candidate action,
ask an analyser, say one sentence about the answer.* This module holds that pair
of records so the control flow is written once per app instead of once per
question type.

``Ask.snapshot`` is deliberately a plain string (a FEN, a word, whatever the app
serialises to). Nothing here knows where it came from, which is what lets a
position arrive from the live game, the browser, a puzzle file, or a camera
without changing the question layer.
"""

from dataclasses import dataclass, field
from typing import Any

#: Who raised the question. ``kid`` = the user asked out loud; ``guard`` = the
#: app raised it before committing the user's action; ``drill`` = it is part of a
#: scripted exercise.
ORIGINS = ("kid", "guard", "drill")


@dataclass(frozen=True)
class Ask:
    """A question about one snapshot of app state.

    Attributes:
        kind: Question type, e.g. ``best`` / ``hint`` / ``whatif`` / ``why`` /
            ``check`` / ``explain``. Selects the prompt, not the code path.
        snapshot: The state to reason about, serialised by the app (never the
            live mutable state).
        action: Candidate action to consider, in the app's own notation. Empty
            when the question is about the snapshot as it stands.
        origin: One of :data:`ORIGINS`; drives budgets and wording.
        request_id: Correlation id for the analyser effect.
        extra: App-specific fields (e.g. chess ``side``) kept out of core.
    """

    kind: str
    snapshot: str
    action: str = ""
    origin: str = "kid"
    request_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Verdict:
    """What the analyser said about an :class:`Ask`.

    Attributes:
        score: How good the snapshot is for the side/user in question, in the
            app's own units (centipawns for chess).
        best_action: The action the analyser recommends.
        delta: How much worse ``Ask.action`` is than ``best_action``; 0 when no
            candidate action was given. Positive means worse.
        severity: Coarse band derived from ``delta``: ``fine`` / ``inaccuracy`` /
            ``mistake`` / ``blunder``.
        detail: App-specific analyser output (lines, mate distance, ...).
    """

    score: int = 0
    best_action: str = ""
    delta: int = 0
    severity: str = "fine"
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Refusal:
    """Why a question cannot be answered, in words the app can speak.

    Attributes:
        reason: Machine-readable tag, e.g. ``illegal_action``.
        message: One sentence to say out loud.
    """

    reason: str
    message: str


def severity_for(delta: int, thresholds: dict[str, int]) -> str:
    """Band a score loss into a severity name.

    Args:
        delta: How much the candidate action loses, in app units.
        thresholds: Mapping of severity name to its minimum ``delta``, e.g.
            ``{"inaccuracy": 50, "mistake": 100, "blunder": 200}``.

    Returns:
        The name of the highest band whose threshold ``delta`` reaches, or
        ``"fine"`` when it reaches none.
    """
    band = "fine"
    for name, minimum in sorted(thresholds.items(), key=lambda item: item[1]):
        if delta >= minimum:
            band = name
    return band
