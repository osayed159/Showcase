import unittest

from answer import unsupported_numbers

PASSAGES = [
    {"text": "Our model reaches a CER of 4.02 on IAM, evaluated in 2013 on 11,273 images."},
    {"text": "A second passage that mentions 9.99 only here."},
    {"text": "built on ResNet18 backbones"},
]


def flagged(sentence: str) -> list[str]:
    return [number for number, _ in unsupported_numbers(sentence, PASSAGES)]


class UnsupportedNumbersTest(unittest.TestCase):
    def test_number_in_cited_passage_passes(self):
        self.assertEqual(flagged("The CER is 4.02 [1]."), [])

    def test_invented_number_is_flagged(self):
        self.assertEqual(flagged("The CER is 7.3 [1]."), ["7.3"])

    def test_number_hidden_inside_a_longer_number_is_flagged(self):
        self.assertEqual(flagged("It was trained for 3 epochs [1]."), ["3"])
        self.assertEqual(flagged("The CER is 4.0 [1]."), ["4.0"])

    def test_thousands_separators(self):
        self.assertEqual(flagged("It used 11,273 images [1]."), [])
        self.assertEqual(flagged("It used 273 images [1]."), ["273"])

    def test_number_from_a_different_passage_than_cited_is_flagged(self):
        self.assertEqual(flagged("The CER is 4.02 [2]."), ["4.02"])

    def test_uncited_sentence_is_checked_against_all_passages(self):
        self.assertEqual(flagged("Another figure is 9.99."), [])
        self.assertEqual(flagged("The CER is 1.23 on IAM."), ["1.23"])

    def test_citation_after_the_full_stop_counts_for_its_sentence(self):
        self.assertEqual(flagged("The CER is 4.02 on IAM. [1]"), [])
        self.assertEqual(flagged("The CER is 1.23 on IAM. [1]"), ["1.23"])

    def test_number_glued_to_letters_in_the_passage(self):
        self.assertEqual(flagged("It uses ResNet 18 [3]."), [])


if __name__ == "__main__":
    unittest.main()
