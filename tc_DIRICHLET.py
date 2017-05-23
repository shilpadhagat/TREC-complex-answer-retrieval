import math

"""
DIRICHLET SMOOTHING ALGORITHM

@author: Gaurav Patil.
"""

class DIRICHLET:

    useCache = False
    number_of_words_in_the_collection_s = 0
    all_words_freq_dict = dict()

    def __init__(self, query_structure, document_structure, tuning_parameter):
        """
        Constructor takes the query structure and the document structure
        :param query_structure: tuple (query_id_plain, query_id_formatted, Ranked dict of words))
        :param document_structure: dictionary  consisting of document id mapping to the ranked dict of words
        :param tuning_parameter: 2500 for 500,000 documents 
        """
        self.queries = query_structure
        self.documents = document_structure
        self.frequency_of_all_words_in_a_collection = self.freq_of_all_words()
        self.no_of_words_in_the_collection = self.no_of_words_in_collection()
        self.u = tuning_parameter

    def no_of_words_in_collection(self):
        """
        Total no of words in the collection
        :return: int (total no of words in all the paragraphs)
        """
        if DIRICHLET.useCache:
            return DIRICHLET.number_of_words_in_the_collection_s
        else:
            summ = 0
            for para_id, ranked_words_dict in self.documents.items():
                summ += sum(ranked_words_dict.values())
            for elem in self.queries:
                summ += sum(elem[2].values())
            return summ

    def freq_of_all_words(self):
        """
        Generates a ranked list of frequency of all words in the collection
        :return: dict ranked dictionary
        """
        if DIRICHLET.useCache:
            return DIRICHLET.all_words_freq_dict
        else:
            freq_dict = dict()
            for para_id, ranked_words_dict in self.documents.items():
                for word, frequency in ranked_words_dict.items():
                    if word in freq_dict:
                        freq_dict[word] += frequency
                    else:
                        freq_dict[word] = frequency
            for elem in self.queries:
                for k, freq in elem[2].items():
                    if k in freq_dict:
                        freq_dict[k] += freq
                    else:
                        freq_dict[k] = freq
            return freq_dict

    def word_frequency_of_word_in_document(self, word, document_id):
        """
        Returns the frequency of word in the document
        :param word: query term
        :param document_id: document id
        :return: frequency of the word in the document
        """
        ranked_words_dict = self.documents[document_id]
        if word in ranked_words_dict:
            return ranked_words_dict[word]
        else:
            return 0

    def score(self, query, document_id):
        """
        Returns the score given a query and a document
        :param query: query tup structure
        :param document_id: document id 
        :return: tup (query, document_id and score)
        """
        score = 0
        document_length = sum(self.documents[document_id].values())
        query_length = sum(query[2].values())
        part_one_calc = query_length * math.log(self.u / (document_length + self.u))
        for key, value in query[2].items():
            w = self.word_frequency_of_word_in_document(key, document_id)
            inner_calc = ((w / self.u) * (self.no_of_words_in_the_collection / self.frequency_of_all_words_in_a_collection[key])) + 1.0
            score += query[2][key] * math.log(inner_calc)
        score += part_one_calc
        tup = (query[1], document_id, score)
        return tup
