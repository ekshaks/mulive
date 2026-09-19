import asyncio
import unittest

from reactivex.subject import Subject

from mulive.core.async_controller_flow import AsyncControllerFlow
from mulive.core.stream_dsl import Stream, SubGroup


class EchoWorkflow:
    async def run(self, ctx):
        ctx.emit("ready")
        first = await ctx.wait_for(str)
        ctx.emit(first.upper())


class PredicateWorkflow:
    async def run(self, ctx):
        event = await ctx.wait_for(lambda item: item == "target")
        ctx.emit(event)


class NextEventWorkflow:
    async def run(self, ctx):
        first = await ctx.next_event()
        second = await ctx.next_event()
        ctx.emit((first, second))


class ErrorWorkflow:
    async def run(self, ctx):
        raise ValueError("bad workflow")


class AsyncControllerFlowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.input_subject = Subject()
        self.subs = SubGroup()

    async def asyncTearDown(self):
        self.subs.dispose()

    def attach_input(self, flow):
        Stream.source(self.input_subject).to(
            flow.input_sink(),
            subs=self.subs,
        )

    async def test_workflow_emits_and_waits_for_events(self):
        flow = AsyncControllerFlow(EchoWorkflow(), name="test_async_controller")
        self.attach_input(flow)
        received = []
        flow.outputs.to(
            lambda observable: observable.subscribe(received.append),
            subs=self.subs,
        )

        flow.start()
        await asyncio.sleep(0)
        self.input_subject.on_next("hello")
        await asyncio.sleep(0.01)

        self.assertEqual(received, ["ready", "HELLO"])

    async def test_submit_before_start_is_buffered(self):
        flow = AsyncControllerFlow(EchoWorkflow(), name="test_async_controller")
        self.attach_input(flow)
        received = []
        flow.outputs.to(
            lambda observable: observable.subscribe(received.append),
            subs=self.subs,
        )

        self.assertTrue(flow.submit("early"))
        flow.start()
        await asyncio.sleep(0.01)

        self.assertEqual(received, ["ready", "EARLY"])

    async def test_wait_for_accepts_predicate(self):
        flow = AsyncControllerFlow(PredicateWorkflow(), name="test_async_controller")
        self.attach_input(flow)
        received = []
        flow.outputs.to(
            lambda observable: observable.subscribe(received.append),
            subs=self.subs,
        )

        flow.start()
        flow.submit("ignored")
        flow.submit("target")
        await asyncio.sleep(0.01)

        self.assertEqual(received, ["target"])

    async def test_next_event_returns_events_without_filtering(self):
        flow = AsyncControllerFlow(NextEventWorkflow(), name="test_async_controller")
        self.attach_input(flow)
        received = []
        flow.outputs.to(
            lambda observable: observable.subscribe(received.append),
            subs=self.subs,
        )

        flow.start()
        flow.submit("ignored")
        flow.submit("target")
        await asyncio.sleep(0.01)

        self.assertEqual(received, [("ignored", "target")])

    async def test_close_completes_outputs_and_rejects_late_submit(self):
        flow = AsyncControllerFlow(EchoWorkflow(), name="test_async_controller")
        self.attach_input(flow)
        completed = []
        flow.outputs.to(
            lambda observable: observable.subscribe(
                on_completed=lambda: completed.append(True)
            ),
            subs=self.subs,
        )

        flow.start()
        flow.close()

        self.assertEqual(completed, [True])
        self.assertFalse(flow.submit("late"))

    async def test_workflow_error_propagates(self):
        flow = AsyncControllerFlow(ErrorWorkflow(), name="test_async_controller")
        self.attach_input(flow)
        errors = []
        flow.outputs.to(
            lambda observable: observable.subscribe(on_error=errors.append),
            subs=self.subs,
        )

        flow.start()
        await asyncio.sleep(0.01)

        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ValueError)


if __name__ == "__main__":
    unittest.main()
