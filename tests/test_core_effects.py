import asyncio
import json
import unittest
from dataclasses import dataclass

from reactivex.subject import Subject

from mulive.apps.app_output import AppOutput, output
from mulive.apps.effects import EffectRunner
from mulive.apps.events import FeedbackEvent
from mulive.apps.prompts import (
    extract_json_object,
    load_prompt_instructions,
    load_prompt_request,
)
from mulive.apps.qa import Ask, Refusal, Verdict, severity_for
from mulive.core.stream_dsl import Stream, SubGroup


@dataclass(frozen=True)
class RequestThing:
    request_id: str
    payload: str = ""


@dataclass(frozen=True)
class ThingDone:
    request_id: str
    value: str


@dataclass(frozen=True)
class ThingFailed:
    request_id: str
    reason: str


@dataclass(frozen=True)
class UnknownRequest:
    request_id: str


class EffectRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.submitted = []
        self.runner = EffectRunner(self.submitted.append, name="test_effects")

    async def asyncTearDown(self):
        await self.runner.close()

    async def test_handler_result_is_submitted(self):
        async def run(request):
            await asyncio.sleep(0)
            return ThingDone(request.request_id, request.payload.upper())

        self.runner.register(RequestThing, run)
        task = self.runner.start(RequestThing("r1", "hello"))
        await task
        self.assertEqual(self.submitted, [ThingDone("r1", "HELLO")])

    async def test_handler_returning_none_submits_nothing(self):
        async def run(request):
            return None

        self.runner.register(RequestThing, run)
        await self.runner.start(RequestThing("r1"))
        self.assertEqual(self.submitted, [])

    async def test_error_handler_turns_exception_into_event(self):
        async def run(request):
            raise RuntimeError("engine died")

        self.runner.register(
            RequestThing,
            run,
            on_error=lambda request, exc: ThingFailed(request.request_id, str(exc)),
        )
        await self.runner.start(RequestThing("r2"))
        self.assertEqual(self.submitted, [ThingFailed("r2", "engine died")])

    async def test_exception_without_error_handler_is_dropped(self):
        async def run(request):
            raise RuntimeError("boom")

        self.runner.register(RequestThing, run)
        await self.runner.start(RequestThing("r3"))
        self.assertEqual(self.submitted, [])

    async def test_none_and_unregistered_requests_are_ignored(self):
        async def run(request):
            return ThingDone(request.request_id, "x")

        self.runner.register(RequestThing, run)
        self.assertIsNone(self.runner.start(None))
        self.assertIsNone(self.runner.start(UnknownRequest("r4")))
        self.assertEqual(self.submitted, [])

    async def test_duplicate_registration_rejected(self):
        async def run(request):
            return None

        self.runner.register(RequestThing, run)
        with self.assertRaises(ValueError):
            self.runner.register(RequestThing, run)

    async def test_close_cancels_outstanding_effects(self):
        started = asyncio.Event()

        async def run(request):
            started.set()
            await asyncio.sleep(10)
            return ThingDone(request.request_id, "late")

        self.runner.register(RequestThing, run)
        task = self.runner.start(RequestThing("r5"))
        await started.wait()
        await self.runner.close()
        self.assertTrue(task.cancelled())
        self.assertEqual(self.submitted, [])

    async def test_start_after_close_is_ignored(self):
        async def run(request):
            return ThingDone(request.request_id, "x")

        self.runner.register(RequestThing, run)
        await self.runner.close()
        self.assertIsNone(self.runner.start(RequestThing("r6")))

    async def test_concurrent_effects_all_complete(self):
        async def run(request):
            await asyncio.sleep(0.01)
            return ThingDone(request.request_id, request.payload)

        self.runner.register(RequestThing, run)
        tasks = [self.runner.start(RequestThing(f"r{i}", str(i))) for i in range(5)]
        await asyncio.gather(*tasks)
        self.assertEqual(
            sorted(event.request_id for event in self.submitted),
            ["r0", "r1", "r2", "r3", "r4"],
        )

    async def test_sink_runs_effects_from_an_output_stream(self):
        async def run(request):
            return ThingDone(request.request_id, "from-stream")

        self.runner.register(RequestThing, run)
        subs = SubGroup()
        subject = Subject()
        Stream.source(subject, name="outputs").to(self.runner.sink(), subs=subs)
        try:
            subject.on_next(output(None, "no effect here"))
            subject.on_next(output(None, "thinking", effect=RequestThing("r7")))
            await asyncio.sleep(0.01)
        finally:
            subs.dispose()
        self.assertEqual(self.submitted, [ThingDone("r7", "from-stream")])


class AppOutputTests(unittest.TestCase):
    def test_empty_messages_are_dropped(self):
        packet = output("state", "said something", "", None)
        self.assertEqual(packet.messages, ["said something"])
        self.assertEqual(packet.state, "state")
        self.assertFalse(packet.finished)
        self.assertIsNone(packet.effect)
        self.assertIsNone(packet.feedback)

    def test_all_fields_pass_through(self):
        feedback = FeedbackEvent("chess_position", "update", {"fen": "start"})
        packet = output(
            "state",
            "hi",
            finished=True,
            effect=RequestThing("r1"),
            feedback=feedback,
        )
        self.assertTrue(packet.finished)
        self.assertEqual(packet.effect, RequestThing("r1"))
        self.assertEqual(packet.feedback, feedback)

    def test_default_output_is_empty(self):
        self.assertEqual(AppOutput().messages, [])


class PromptsTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "prompts.yml"
        self.path.write_text(
            "coach_best:\n"
            "  instructions: |\n"
            "    Be kind.\n"
            "  request: |\n"
            "    Position: {fen}\n"
            "empty_one:\n"
            "  instructions: ''\n"
        )

    def tearDown(self):
        self.dir.cleanup()

    def test_load_both_sections(self):
        self.assertEqual(load_prompt_instructions(self.path, "coach_best"), "Be kind.")
        self.assertEqual(load_prompt_request(self.path, "coach_best"), "Position: {fen}")

    def test_missing_prompt_raises(self):
        with self.assertRaises(ValueError):
            load_prompt_instructions(self.path, "nope")

    def test_empty_section_raises(self):
        with self.assertRaises(ValueError):
            load_prompt_instructions(self.path, "empty_one")
        with self.assertRaises(ValueError):
            load_prompt_request(self.path, "coach_best".replace("best", "missing"))

    def test_extract_json_object_variants(self):
        self.assertEqual(extract_json_object('{"a": 1}'), {"a": 1})
        self.assertEqual(extract_json_object('```json\n{"a": 2}\n```'), {"a": 2})
        self.assertEqual(extract_json_object('sure! {"a": 3} done'), {"a": 3})
        self.assertEqual(extract_json_object("no json here"), {})
        self.assertEqual(extract_json_object("[1, 2]"), {})
        self.assertEqual(extract_json_object("{not json}"), {})

    def test_extract_json_object_strict(self):
        with self.assertRaises(ValueError):
            extract_json_object("no json here", strict=True)
        with self.assertRaises(ValueError):
            extract_json_object("[1, 2]", strict=True)
        with self.assertRaises(json.JSONDecodeError):
            extract_json_object("{not json}", strict=True)


class QaTests(unittest.TestCase):
    def test_ask_defaults(self):
        ask = Ask(kind="best", snapshot="fen")
        self.assertEqual(ask.action, "")
        self.assertEqual(ask.origin, "kid")
        self.assertEqual(ask.extra, {})

    def test_verdict_defaults(self):
        verdict = Verdict()
        self.assertEqual((verdict.score, verdict.delta, verdict.severity), (0, 0, "fine"))
        self.assertEqual(verdict.best_action, "")

    def test_refusal_carries_a_spoken_message(self):
        refusal = Refusal("illegal_action", "That move is not legal.")
        self.assertEqual(refusal.message, "That move is not legal.")

    def test_severity_bands(self):
        thresholds = {"inaccuracy": 50, "mistake": 100, "blunder": 200}
        self.assertEqual(severity_for(0, thresholds), "fine")
        self.assertEqual(severity_for(49, thresholds), "fine")
        self.assertEqual(severity_for(50, thresholds), "inaccuracy")
        self.assertEqual(severity_for(150, thresholds), "mistake")
        self.assertEqual(severity_for(900, thresholds), "blunder")
        self.assertEqual(severity_for(900, {}), "fine")


if __name__ == "__main__":
    unittest.main()
