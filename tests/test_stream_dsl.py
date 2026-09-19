import unittest

from reactivex.subject import Subject

from mulive.core.stream_dsl import Stream, map_filter_items


class MapFilterItemsTests(unittest.TestCase):
    def test_maps_before_filtering(self):
        source = Subject()
        mapped = []
        received = []

        stream = Stream.source(source) | map_filter_items(
            map_fn=lambda value: mapped.append(value) or value * 2,
            filter_fn=lambda value: value > 2,
        )
        subscription = stream.observable.subscribe(received.append)

        source.on_next(1)
        source.on_next(2)
        source.on_next(3)

        self.assertEqual(mapped, [1, 2, 3])
        self.assertEqual(received, [4, 6])
        subscription.dispose()


if __name__ == "__main__":
    unittest.main()
