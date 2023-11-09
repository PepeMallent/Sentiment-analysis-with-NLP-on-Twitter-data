import unittest
from unittest import TestCase
from unittest.mock import patch
from io import StringIO

from twitter_processor import twitter_processor


class TestTwitterProcessor(TestCase):
    def test_load_data(self):
        with patch('builtins.open', unittest.mock.mock_open(
                    read_data='col1,col2,col3\nval1,val2,val3')):
            dataset = twitter_processor.load_data(file='test.csv')
            self.assertEqual(dataset, [{'col1': 'val1', 'col2':
                                        'val2', 'col3': 'val3'}])

    def test_print_data(self):
        dataset = [{'col1': 'val1', 'col2': 'val2', 'col3': 'val3'}]
        expected_output = "{'col1': 'val1', 'col2': 'val2', 'col3': 'val3'}\n"
        with patch('sys.stdout', new=StringIO()) as fake_output:
            twitter_processor.print_data(dataset)
            self.assertEqual(fake_output.getvalue(), expected_output)

    def test_preprocess_text(self):
        dataset = [{'text': 'Hello, www.example.com @username #hashtag'}]
        expected_output = [{'text': 'hello   hashtag'}]
        twitter_processor.preprocess_text(dataset)
        self.assertEqual(dataset, expected_output)

    def test_remove_stopwords(self):
        dataset = [{'text': 'me myself love this python programming'},
                   {'text': 'am is are this example because '
                            'sentence have them'}]
        expected_output = [{'text': 'love python programming'},
                           {'text': 'example sentence'}]
        twitter_processor.remove_stopwords(dataset)
        self.assertEqual(dataset, expected_output)

    def test_get_term_frequencies(self):
        dataset = [{'text': 'this is a test tweet'},
                   {'text': 'another example tweet'},
                   {'text': 'this is just a tweet'}]
        expected_output = [{'this': 1, 'is': 1, 'a': 1, 'test': 1, 'tweet': 1},
                           {'another': 1, 'example': 1, 'tweet': 1},
                           {'this': 1, 'is': 1, 'just': 1, 'a': 1, 'tweet': 1}]
        term_frequencies = twitter_processor.get_term_frequencies(dataset)
        self.assertEqual(term_frequencies, expected_output)

    def test_get_vocabulary(self):
        dataset = [{'text': 'this is a test tweet'},
                   {'text': 'another example tweet'},
                   {'text': 'this is just a tweet'}]
        expected_output = ['this', 'is', 'a', 'test', 'tweet',
                           'another', 'example', 'just']
        vocabulary = twitter_processor.get_vocabulary(dataset)
        self.assertEqual(vocabulary.sort(), expected_output.sort())

    def test_print_sorted_list(self):
        words_list = ['zebra', 'apple', 'banana', 'cat', 'dog']
        expected_output = "apple\nbanana\ncat\ndog\nzebra\n"
        with patch('sys.stdout', new=StringIO()) as fake_output:
            twitter_processor.print_sorted_list(words_list)
            self.assertEqual(fake_output.getvalue(), expected_output)

    def test_add_term_frequency_col(self):
        dataset = [{'col1': 'val1'}, {'col1': 'val2'}, {'col1': 'val3'}]
        term_frequencies = [{'term1': 1, 'term2': 2}, {'term1': 3, 'term2': 4},
                            {'term1': 5, 'term2': 6}]
        expected_output = [{'col1': 'val1', 'term_frequency': {'term1': 1,
                            'term2': 2}},
                           {'col1': 'val2', 'term_frequency': {'term1': 3,
                            'term2': 4}},
                           {'col1': 'val3', 'term_frequency': {'term1': 5,
                            'term2': 6}}]
        twitter_processor.add_term_frequency_col(dataset, term_frequencies)
        self.assertEqual(dataset, expected_output)

    def test_find_num_clusters(self):
        dataset = [{'sentiment': 'positive'}, {'sentiment': 'negative'},
                   {'sentiment': 'neutral'}]
        expected_output = "\nThe dataset has 3 clusters in the sentiment " \
                          "column."
        with patch('sys.stdout', new=StringIO()) as fake_output:
            twitter_processor.find_num_clusters(dataset)
            self.assertEqual(fake_output.getvalue().strip(),
                             expected_output.strip())

    def test_find_empty_percentage(self):
        dataset = [{'text': 'this is a tweet'}, {'text': ''},
                   {'text': 'another tweet'}, {'text': ''}]
        expected_output = "\nThere are empty elements in the text\nThe " \
                          "percentage of empty elements in the text " \
                          "column is: 50.00%\n"
        with patch('sys.stdout', new=StringIO()) as fake_output:
            twitter_processor.find_empty_percentage(dataset)
            self.assertEqual(fake_output.getvalue(), expected_output)

    def test_eliminate_null_elements(self):
        dataset = [{'text': 'this is a tweet'}, {'text': ''},
                   {'text': 'another tweet'}, {'text': ''}]
        expected_output = [{'text': 'this is a tweet'},
                           {'text': 'another tweet'}]
        dataset = twitter_processor.eliminate_null_elements(dataset)
        self.assertEqual(dataset, expected_output)


if __name__ == '__main__':
    unittest.main()
