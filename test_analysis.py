import unittest
from analysis import parse_line

class TestAnalysis(unittest.TestCase):

    def test_parse_line_valid(self):
        line = "1,7.0,6.5,8.0,9.0,N1,8.8,7.5,6.0,8.0,7.5,12345678"
        sbd, scores = parse_line(line)
        self.assertEqual(sbd, "12345678")
        self.assertEqual(len(scores), 10)
        self.assertEqual(scores[0], 7.0)
        self.assertIsNone(scores[4]) # Ma_mon_ngoai_ngu should be None
        self.assertEqual(scores[9], 7.5)

    def test_parse_line_with_empty_scores(self):
        line = "2,,,,,,,,,,12345679"
        sbd, scores = parse_line(line)
        self.assertEqual(sbd, "12345679")
        self.assertTrue(all(s is None for s in scores))

    def test_parse_line_invalid(self):
        line = "invalid_line"
        sbd, scores = parse_line(line)
        self.assertIsNone(sbd)
        self.assertEqual(scores, [])

if __name__ == '__main__':
    unittest.main()
