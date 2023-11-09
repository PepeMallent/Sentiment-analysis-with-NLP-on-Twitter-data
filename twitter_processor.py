import csv
import zipfile
import re
from collections import defaultdict
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from collections import Counter


def unzip_file(zip_filepath, extract_path):
    """
        Unzip a file and save it to the specified directory.

        Args:
            zip_filepath (str): Path to the ZIP file.
            extract_path (str): Path to the directory where the ZIP
                                file should be extracted.

        Returns:
            str: Full path to the extracted file.
    """
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
        extracted_filename = zip_ref.namelist()[0]
    os.rename(os.path.join(extract_path, extracted_filename),
              os.path.join(extract_path, 'twitter_reduced.csv'))


def load_data(file="data/twitter_reduced.csv"):
    """
        Load data from a CSV file.

        Args:
            file (str): Path to the CSV file.

        Returns:
            list: A list of dictionaries representing the dataset.
    """
    dataset = []  # List to store dictionaries

    with open(file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            dataset.append(row)

    return dataset


def print_data(dataset, num_rows=5, last=False):
    """
        Print the data from the dataset.

        Args:
            dataset (list): The dataset to print.
            num_rows (int): Number of rows to print.
            last (bool): Whether to print the last `num_rows` rows.
    """
    start_idx = 0
    if last:
        start_idx = len(dataset) - num_rows

    for i, data in enumerate(dataset[start_idx:]):
        print(data)
        if i == (num_rows - 1):  # (0 based index)
            break


def preprocess_text(dataset, text_col='text'):
    """
       Preprocess the text data in the dataset.

       Args:
           dataset (list): The dataset to preprocess.
           text_col (str): The name of the column containing the text data.
    """
    for data in dataset:
        # Eliminate URLs
        data[text_col] = re.sub(r'http\S+|www\S+', '', data[text_col])

        # Remove non-ASCII special characters
        data[text_col] = re.sub(r'[^\x00-\x7F]+', '', data[text_col])

        # Remove any words that starts with a symbol
        data[text_col] = re.sub(r'[@;:\']\b\S+\b', '', data[text_col])

        # Remove symbols
        data[text_col] = re.sub(r'[^a-zA-Z0-9\s]', '', data[text_col])

        # Remove numbers
        data[text_col] = re.sub(r'\d+', '', data[text_col])

        # Convert text to lowercase and trim
        data[text_col] = data[text_col].lower().strip()


def remove_stopwords(dataset, text_col='text'):
    """
        Remove stopwords from the text data in the dataset.

        Args:
            dataset (list): The dataset to process.
            text_col (str): The name of the column containing the text data.
    """
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                 'his', 'himself', 'she', 'her',  'hers', 'herself', 'it',
                 'its', 'itself', 'they', 'them', 'their', 'theirs',
                 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
                 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
                 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
                 'with', 'about', 'against', 'between', 'into', 'through',
                 'during', 'before', 'after', 'above', 'below', 'to', 'from',
                 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
                 'can', 'will', 'just', 'don', 'should', 'now']

    for data in dataset:
        text = data[text_col]
        words = text.split()
        filtered_words = [word for word in words if word.lower()
                          not in stopwords]
        processed_text = ' '.join(filtered_words)
        data[text_col] = processed_text


def get_term_frequencies(dataset, text_col='text'):
    """
        Get the term frequencies for each entry in the dataset's text column.

        Args:
            dataset (list): The dataset to process.
            text_col (str): The name of the column containing the text data.

        Returns:
            list: A list of dictionaries representing the term frequencies
                  for each entry.
    """
    term_frequencies = []

    for data in dataset:
        tweet = data[text_col]
        term_frequency = defaultdict(int)

        words = re.findall(r'\b\w+\b', tweet)
        for word in words:
            term_frequency[word] += 1

        term_frequencies.append(dict(term_frequency))

    return term_frequencies


def get_vocabulary(dataset, text_col='text'):
    """
        Get the vocabulary (unique words) in the dataset.

        Args:
            dataset (list): The dataset to process.
            text_col (str): The name of the column containing the text data.

        Returns:
            list: A list of unique words in the dataset.
    """
    vocabulary = set()
    for data in dataset:
        text = data[text_col]
        words = text.split()
        vocabulary.update(words)
    return list(vocabulary)


def print_sorted_list(words_list, count=10):
    """
        Sort and print the list of words.

        Args:
            words_list (list): The list of words.
            count (int): Number of words to print.
    """
    sorted_words = sorted(words_list)
    for word in sorted_words[:count]:
        print(word)


def add_term_frequency_col(dataset, term_frequencies,
                           col_name='term_frequency'):
    """
        Add a column to the dataset containing the term frequencies.

        Args:
            dataset (list): The dataset to modify.
            term_frequencies (list): The list of term frequencies.
            col_name (str): The name of the new column.
    """

    for i, data in enumerate(dataset):
        data[col_name] = term_frequencies[i]


def write_to_csv(dataset, output_file='data/twitter_processed.csv'):
    """
        Write the dataset to a CSV file.

        Args:
            dataset (list): The dataset to write.
            output_file (str): Path to the output CSV file.
    """

    # Extract the keys from the first record to use as column headers
    headers = dataset[0].keys()

    # Write the dataset to a CSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(dataset)

    print(f"\nProcessed dataset saved to {output_file}")


def find_num_clusters(dataset, col_name='sentiment'):
    """
        Find the number of clusters in a dataset based on a specified column.

        Args:
            dataset (list): The dataset to analyze.
            col_name (str): The name of the column containing the cluster
                            information.

    """
    sentiment_values = set(data[col_name] for data in dataset)
    num_clusters = len(sentiment_values)
    print(f"\nThe dataset has {num_clusters} clusters in "
          f"the {col_name} column.")


def find_empty_percentage(dataset, text_col='text'):
    """
        Find the percentage of empty elements in a specific column of
        a dataset.

        Args:
            dataset (list): The dataset to analyze.
            text_col (str): The name of the column to check for empty elements.

    """

    empty = 0

    for data in dataset:
        if not data[text_col]:
            empty += 1

    if empty > 0:
        print(f"\nThere are empty elements in the {text_col}")
        empty_percentage = (empty / len(dataset)) * 100
        print(f"The percentage of empty elements in the {text_col} "
              f"column is: {empty_percentage:.2f}%")

    else:
        print(f"There are no empty elements in the {text_col}")


def eliminate_null_elements(dataset, text_col='text'):
    """
        Eliminate records with null or empty values in a specific column from
        a dataset.

        Args:
            dataset (list): The dataset to process.
            text_col (str): The name of the column to check for null or empty
            values.

        Returns:
            list: The updated dataset with null or empty records removed.
    """

    dataset = [data for data in dataset if data[text_col]]
    return dataset


def generate_word_clouds_for_clusters(dataset, cluster_col='sentiment',
                                      text_col='text',
                                      output_dir='word_cloud_plots'):
    """
        Generate word clouds for each cluster in a dataset based on a
        specified column.

        Args:
            dataset (list): The dataset containing the records.
            cluster_col (str): The name of the column representing the
                               clusters.
            text_col (str): The name of the column containing the text data.
            output_dir (str): The directory to save the generated word cloud
                              plots.

    """
    cluster_data = {}

    # group records by cluster
    for data in dataset:
        cluster = data[cluster_col]
        text = data[text_col]
        if cluster in cluster_data:
            cluster_data[cluster].append(text)
        else:
            cluster_data[cluster] = [text]

    # generate word cloud for each cluster
    for cluster, records in cluster_data.items():
        print("\nGenerating word cloud for cluster ", cluster)
        combined_text = ' '.join(records)

        # tokenize words
        words = nltk.word_tokenize(combined_text)

        # create word frequency dictionary
        word_freq = nltk.FreqDist(words)

        # generate word cloud
        wordcloud = WordCloud().generate_from_frequencies(word_freq)

        # save word cloud plot
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, 'cluster_{}.png'.format(
                      cluster))
        wordcloud.to_file(output_file)

        # display word cloud
        plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Cluster {}'.format(cluster))
        plt.show()


def get_cluster_term_frequencies(dataset, text_col='text',
                                 cluster_col='sentiment'):
    """
        Calculate term frequencies for each word within clusters in a dataset.

        Args:
            dataset (list): The dataset containing the records.
            text_col (str): The name of the column containing the text data.
            cluster_col (str): The name of the column representing the
                               clusters.

        Returns:
            dict: A nested dictionary containing the term frequencies for each
                  cluster. The outer dictionary's keys are the clusters, and
                  the inner dictionary contains word-frequency pairs for each
                  cluster.
        """

    term_frequencies = defaultdict(dict)

    for data in dataset:
        cluster = data[cluster_col]
        tweet = data[text_col]

        words = re.findall(r'\b\w+\b', tweet)
        for word in words:
            term_frequencies[cluster][word] = term_frequencies[
                                              cluster].get(word, 0) + 1

    return term_frequencies


def generate_cluster_histograms(term_frequencies, output_dir='histograms',
                                top_count=20):
    """
        Generate histograms for the top words in each cluster based on term
        frequencies.

        Args:
            term_frequencies (dict): A nested dictionary containing the term
                                     frequencies for each cluster. The outer
                                     dictionary's keys are the clusters, and
                                     the inner dictionary contains
                                     word-frequency pairs for each cluster.
            output_dir (str): The directory to save the generated histograms.
                              Default is 'histograms'.
            top_count (int): The number of top words to include in the
                             histogram. Default is 20.

    """

    for cluster, frequencies in term_frequencies.items():
        print("\nGenerating histogram for cluster ", cluster)
        word_counts = Counter(frequencies)
        top_words = word_counts.most_common(top_count)
        words, counts = zip(*top_words)

        plt.bar(words, counts)
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.title(f'Top 20 Words - Cluster {cluster}')
        plt.xticks(rotation=90)

        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/histogram_cluster{cluster}.png')

        plt.show()
        plt.clf()
