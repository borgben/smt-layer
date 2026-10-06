import unittest
import z3
import test_alternative_conflicts as existing_tests


class SingletonCacheTests(unittest.TestCase):
    def test_instance_polarity_context_and_set_recurrence(self):
        tracker = existing_tests.module.SingletonCacheDiagnostic()
        label, bit, output = z3.Bools('var0 bit output')
        probe = ([label], {'singleton_count': 1})
        tracker.observe(0, [output], [(label, bit)], probe)
        tracker.observe(1, [output], [(label, bit)], probe)
        self.assertEqual(tracker.summary()['singleton_hits'], 0)
        tracker.observe(0, [z3.Not(output)], [(label, bit)], probe)
        self.assertEqual(tracker.summary()['singleton_hits'], 1)
        self.assertEqual(tracker.summary()['same_context_hits'], 0)
        tracker.observe(0, [output], [(label, bit)], probe)
        tracker.observe(0, [output], [(label, z3.Not(bit))], probe)
        report = tracker.summary()
        self.assertEqual(report['singleton_hits'], 2)
        self.assertEqual(report['same_context_hits'], 1)
        self.assertEqual(report['whole_set_hits'], 2)
        self.assertEqual(report['unique_singletons'], 3)

    def test_empty_and_inconclusive_are_not_hits(self):
        tracker = existing_tests.module.SingletonCacheDiagnostic()
        tracker.observe(0, [], [], ([], {'singleton_count': None}))
        tracker.observe(0, [], [], ([], {'singleton_count': 0}))
        tracker.observe(0, [], [], ([], {'singleton_count': 0}))
        report = tracker.summary()
        self.assertEqual(report['inconclusive_queries'], 1)
        self.assertEqual(report['whole_set_hits'], 0)
        self.assertIsNone(report['singleton_hit_rate'])
