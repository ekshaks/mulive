import unittest

from reactivex.subject import Subject

from mulive.core.events import TranscriptEvent
from mulive.core.stream_dsl import Stream, SubGroup, final_transcript_text


class TranscriptEventTests(unittest.TestCase):
    def test_final_text_filters_partial_and_empty_events(self):
        source = Subject()
        received = []
        subs = SubGroup()
        final_text = Stream.source(source) | final_transcript_text()
        final_text.to(
            lambda observable: observable.subscribe(received.append),
            subs=subs,
        )

        source.on_next(TranscriptEvent("forty", is_final=False))
        source.on_next(TranscriptEvent("", is_final=True))
        source.on_next(TranscriptEvent("forty two", is_final=True))

        self.assertEqual(received, ["forty two"])
        subs.dispose()


if __name__ == "__main__":
    unittest.main()
