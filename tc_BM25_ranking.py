import math

"""
Basic BM25
@author: Gaurav Patil
"""


class BM25:
    useCache = False
    no_of_docs_dict = dict()
    average_doc_length = 0.0

    def __init__(self, query_structure, document_structure):
        """
        Constructor takes the query structure and the document structure
        :param query_structure: tuple (query_id_plain, query_id_formatted, Ranked dict of words))
        :param document_structure: dictionary  consisting of document id mapping to the ranked dict of words
        """
        self.queries = query_structure
        self.documents = document_structure
        self.no_of_documents = len(self.documents.keys())
        self.average_length_of_all_documents = self.average_length_of_documents()
        self.k = 1.2
        self.b = 0.75
        self.k_plus_one = self.k + 1
        self.cache = dict()

    def average_length_of_documents(self):
        """
        Calculates the the average length of documents
        :return: average length of documents
        """
        if BM25.useCache:
            return BM25.average_doc_length
        else:
            summ = 0
            for key, value in self.documents.items():
                for k, v in value.items():
                    summ += v
            return summ / float(self.no_of_documents)

    def inverse_document_frequency(self, query_word):
        """
        Takes the query term and returns the inverse document for a query term
        :param query_word: string
        :return: float: idf value
        """
        no_qi = self.no_of_documents_containing_a_word(query_word)
        return float(math.log(self.no_of_documents / (no_qi + 1.0)))

    def no_of_documents_containing_a_word(self, query_word):
        """
        Given a query term returns the no of documents containing the word
        :param query_word: 
        :return: 
        """
        if BM25.useCache:
            if query_word in BM25.no_of_docs_dict:
                return float(BM25.no_of_docs_dict[query_word])
            else:
                return 0
        else:
            if query_word in self.cache:
                return float(self.cache[query_word])
            else:
                no_of_documents_having_the_word = 0
                for para_id, ranked_word_dict in self.documents.items():
                    if query_word in ranked_word_dict:
                        no_of_documents_having_the_word += 1
                self.cache[query_word] = no_of_documents_having_the_word
                return float(no_of_documents_having_the_word)


    def word_frequency_of_word_in_document(self, word, document_id):
        """
        Finds the frequency of a word in the document
        :param word: string
        :param document_id: string
        :return: int occurrence of the word
        """
        if word in self.documents[document_id]:
            return self.documents[document_id][word]
        else:
            return 0

    def score(self, query, document_id):
        """
        Given a query and a document calculates the score
        :param query: tuple
        :param document_id: string
        :return: tuple (formatted_query, document_id, score) 
        """
        score = 0
        document_length = 0
        for k, v in self.documents[document_id].items():
            document_length += v
        for key, value in query[2].items():
            term_freq = self.word_frequency_of_word_in_document(key, document_id)
            score += self.inverse_document_frequency(key) * (self.k_plus_one * term_freq) / \
                     (self.k * (1.0 - self.b + self.b * (document_length / self.average_length_of_all_documents))
                      + term_freq)
        tup = (query, document_id, score)
        return tup