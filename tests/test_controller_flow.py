import unittest

from reactivex.subject import Subject

from mulive.core.controller_flow import ControllerFlow
from mulive.core.stream_dsl import Stream, SubGroup


class FakeController:
    def __init__(self):
        self.handled = []

    def start(self):
        return "ready"

    def handle_input(self, item):
        self.handled.append(item)
        return item.upper()


class ControllerFlowTests(unittest.TestCase):
    def setUp(self):
        self.input_subject = Subject()
        self.subs = SubGroup()
        self.controller = FakeController()
        self.flow = ControllerFlow(self.controller, name="test_controller")
        Stream.source(self.input_subject).to(
            self.flow.input_sink(),
            subs=self.subs,
        )

    def tearDown(self):
        self.flow.close()
        self.subs.dispose()

    def test_start_emits_initial_output_once(self):
        received = []
        self.flow.outputs.to(
            lambda observable: observable.subscribe(received.append),
            subs=self.subs,
        )

        self.flow.start()
        self.flow.start()

        self.assertEqual(received, ["ready"])

    def test_input_is_handled_once_and_output_fans_out(self):
        first = []
        second = []
        self.flow.outputs.to(
            lambda observable: observable.subscribe(first.append),
            subs=self.subs,
        )
        self.flow.outputs.to(
            lambda observable: observable.subscribe(second.append),
            subs=self.subs,
        )
        self.flow.start()

        self.input_subject.on_next("hello")

        self.assertEqual(self.controller.handled, ["hello"])
        self.assertEqual(first, ["ready", "HELLO"])
        self.assertEqual(second, ["ready", "HELLO"])

    def test_close_completes_outputs(self):
        completed = []
        self.flow.outputs.to(
            lambda observable: observable.subscribe(
                on_completed=lambda: completed.append(True)
            ),
            subs=self.subs,
        )

        self.flow.close()

        self.assertEqual(completed, [True])

    def test_multiple_input_streams_share_one_controller(self):
        second_input = Subject()
        Stream.source(second_input).to(
            self.flow.input_sink(),
            subs=self.subs,
        )
        self.flow.start()

        self.input_subject.on_next("first")
        second_input.on_next("second")

        self.assertEqual(self.controller.handled, ["first", "second"])

    def test_input_completion_does_not_close_other_inputs(self):
        second_input = Subject()
        Stream.source(second_input).to(
            self.flow.input_sink(),
            subs=self.subs,
        )
        received = []
        self.flow.outputs.to(
            lambda observable: observable.subscribe(received.append),
            subs=self.subs,
        )
        self.flow.start()

        self.input_subject.on_completed()
        second_input.on_next("still active")

        self.assertEqual(received, ["ready", "STILL ACTIVE"])

    def test_submit_uses_same_controller_path(self):
        self.flow.start()

        accepted = self.flow.submit("callback")

        self.assertTrue(accepted)
        self.assertEqual(self.controller.handled, ["callback"])

    def test_submit_after_close_is_ignored(self):
        self.flow.close()

        accepted = self.flow.submit("late")

        self.assertFalse(accepted)
        self.assertEqual(self.controller.handled, [])

    def test_reentrant_submit_is_queued_until_current_output_finishes(self):
        received = []

        def receive(output):
            received.append(output)
            if output == "FIRST":
                self.flow.submit("second")

        self.flow.outputs.to(
            lambda observable: observable.subscribe(receive),
            subs=self.subs,
        )
        self.flow.start()

        self.input_subject.on_next("first")

        self.assertEqual(self.controller.handled, ["first", "second"])
        self.assertEqual(received, ["ready", "FIRST", "SECOND"])


if __name__ == "__main__":
    unittest.main()
