import twitter_processor

if __name__ == '__main__':
    print('\nSolución de la PEC4. Se ejecutan todos los ejercicios menos '
          'el 7, que está resuelto aparte.')

    print('\nEjercicio 1.1: Descomprimimos el zip y guardamos el csv en data')
    twitter_processor.unzip_file("data/twitter_reduced.zip", "data/")

    print('\nEjercicio 1.2: Cargamos el dataset con la estructura propuesta '
          ' y cargamos los 5 primeros registros:')
    dataset = twitter_processor.load_data()
    twitter_processor.print_data(dataset)
    twitter_processor.preprocess_text(dataset)
    twitter_processor.remove_stopwords(dataset)
    print('\nEjercicio 2.1 y ejercicio 2.2: Mostramos los últimos 5 registros '
          'después de realizar el preprocesado y eliminar stopwords:')
    twitter_processor.print_data(dataset, last=True)

    term_frequencies = twitter_processor.get_term_frequencies(dataset)
    print('\nEjercicio 3: Frecuencias de términos (comprobamos que '
          'se almacenan correctamente)')
    twitter_processor.print_data(term_frequencies)

    vocabulary_list = twitter_processor.get_vocabulary(dataset)
    print('\nMostramos los 10 primeros resultados del vocabulario '
          'ordenado alfabéticamente.\n')
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
