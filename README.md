# README

This project contains a set of utility functions for analyzing twitter data for natural language processing (NLP) 
related to sentiment analysis. It includes functions for loading data from a CSV file, preprocessing text, removing stopwords, calculating term frequencies, generating word clouds, and generating histograms for word frequencies within clusters.

## Requirements

The code and tests requires the following libraries:

- nltk
- wordcloud
- matplotlib
- unittest
- coverage

Make sure these libraries are installed in your Python environment before running the code using the following command in a terminal.

```commandline
pip install -r requirements.txt
```

Also download the necessary NLTK resources. Open a Python terminal and run the following commands:

```python
import nltk
nltk.download('punkt')
```

## Code Description

The code snippet provides several functions for processing and analyzing a twitter dataset for natural language processing (NLP) 
project related to sentiment analysis:

- `unzip_file(zip_filepath, extract_path)`: Unzip a file and save it to the specified directory and returns the path to the extracted file.
- `load_data(file)`: Loads data from a CSV file and returns a list of dictionaries representing the dataset.
- `print_data(dataset, num_rows, last)`: Prints the data from the dataset, either the first `num_rows` or the last `num_rows` rows.
- `preprocess_text(dataset, text_col)`: Preprocesses the text data in the dataset by eliminating URLs, non-ASCII special characters, words starting with symbols, symbols, and converting text to lowercase.
- `remove_stopwords(dataset, text_col)`: Removes stopwords from the text data in the dataset.
- `get_term_frequencies(dataset, text_col)`: Calculates the term frequencies for each entry in the dataset's text column.
- `get_vocabulary(dataset, text_col)`: Gets the vocabulary (unique words) in the dataset.
- `print_sorted_list(words_list, count)`: Sorts and prints a list of words.
- `add_term_frequency_col(dataset, term_frequencies, col_name)`: Adds a column to the dataset containing the term frequencies.
- `write_to_csv(dataset, output_file)`: Writes the dataset to a CSV file.
- `find_num_clusters(dataset, col_name)`: Finds the number of clusters in the dataset based on a specified column.
- `find_empty_percentage(dataset, text_col)`: Finds the percentage of empty elements in a specific column of the dataset.
- `eliminate_null_elements(dataset, text_col)`: Eliminates records with null or empty values in a specific column from the dataset.
- `generate_word_clouds_for_clusters(dataset, cluster_col, text_col, output_dir)`: Generates word clouds for each cluster in the dataset based on a specified column.
- `get_cluster_term_frequencies(dataset, text_col, cluster_col)`: Calculates term frequencies for each word within clusters in the dataset.
- `generate_cluster_histograms(term_frequencies, output_dir, top_count)`: Generates histograms for the top words in each cluster based on term frequencies.

## Usage

If you want to run the main.py file from the root directory of the project you can run it with the following command:
```
python twitter_processor/main.py
```

To use the code, make sure you have the required libraries installed. Then, you can import the necessary functions and use them with your own dataset.

For example:

```python
import twitter_processor

if __name__ == '__main__':
    print('\nSolución de la PEC4. Se ejecutan todos los ejercicios menos '
          'el 7, que está resuelto aparte.')

    print('\nEjercicio 1.1: Descomprimos el zip y guardamos el csv en data')
    twitter_processor.unzip_file("data/twitter_reduced.zip", "data/")

    print('\nEjercicio 1.2: Cargamos el dataset con la estructura propuesta:')
    dataset = twitter_processor.load_data()
    twitter_processor.print_data(dataset)
    twitter_processor.preprocess_text(dataset)
    twitter_processor.remove_stopwords(dataset)
    print('\nEjercicio 2.1 y ejercicio 2.2: Después de realizar el '
          'preprocesado y eliminar stopwords:')
    twitter_processor.print_data(dataset)

    term_frequencies = twitter_processor.get_term_frequencies(dataset)
    print('\nEjercicio 3: Frecuencias de términos (comprobamos que '
          'se almacenan correctamente)')
    twitter_processor.print_data(term_frequencies)

    vocabulary_list = twitter_processor.get_vocabulary(dataset)
    print('\nMostramos los 10 primeros resultados del vocabulario \n'
          'ordenado alfabéticamente.')
    twitter_processor.print_sorted_list(vocabulary_list)

    twitter_processor.add_term_frequency_col(dataset, term_frequencies)
    print('\nEjercicio 4.1: Mostramos el elemento 20 del dataset:')
    print(dataset[19])

    print('\nEjercicio 4.2: Guardamos el dataset en formato csv:')
    twitter_processor.write_to_csv(dataset)

    print('\nEjercicio 5.1: Número de clusters:')
    twitter_processor.find_num_clusters(dataset)

    print('\nEjercicio 5.2: Respondemos a las cuestiones:')
    twitter_processor.find_empty_percentage(dataset)

    dataset = twitter_processor.eliminate_null_elements(dataset)

    twitter_processor.find_empty_percentage(dataset)

    print('\nEjercicio 5.3: Generamos word cloud para cada cluster')
    twitter_processor.generate_word_clouds_for_clusters(dataset)

    print('\nEjercicio 6: Generamos histogramas')
    cluster_term_frequencies = twitter_processor.get_cluster_term_frequencies(
                               dataset)
    twitter_processor.generate_cluster_histograms(cluster_term_frequencies)

```

## Running Tests

To run the tests for this project, follow the steps below:


1. Make sure you have installed the required dependencies.

2. Open a terminal or command prompt and navigate to the project's root directory.

3. Run the following command to execute the tests:

```commandline
python -m unittest discover tests
```

## Test Coverage

To check the test coverage of the project, you can use the following command in commandline:

```commandline
coverage run -m unittest discover tests
```
After running the above command, you can generate a coverage report using:

```commandline
coverage report -m
```

## License

This code is released under the MIT License.