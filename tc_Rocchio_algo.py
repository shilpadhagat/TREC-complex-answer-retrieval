class RocchioAlgorithm(object):
    """
    @author: Shilpa Dhagat.
    Perform Rocchio algorithm on a set of relevant documents
    Weighting is based on td-idf and forms a new query vector of weightings iteratively.

    """

    def __init__(self, query_structure, paragraph_structure, relevance_judgments, ir):
        dictionary = ir.create_dictionary(paragraph_structure)
        query_vector = ir.create_query_view(query_structure, dictionary)
        query_mod = self.execute_rocchio(dictionary, relevance_judgments, query_vector, 1, .75, .15)
        self.new_query = self.getNewQuery(query_structure, query_mod, dictionary)

    def get_query_vector(self, query_structure, dictionary):
        query_vector = [0] * len(dictionary)
        for word in query_structure.split():
            pos = dictionary[word]
            query_vector[pos] += query_vector[pos]
        return query_vector

    def execute_rocchio(self, dictionary, relevance, queryvector, alpha, beta, gamma):

        relevant_docs = [relevance[i] for i in range(len(relevance)) if relevance[i][1] > 0.1]
        non_relevant_docs = [relevance[i] for i in range(len(relevance)) if
                        relevance[i][1] <= 0.1]  # weights less than or equal to 0.1 are not considered relevant

        term1 = [alpha * i for i in queryvector]

        sum_relevant_docs = 0
        for i in relevant_docs:
            sum_relevant_docs += i[1]

        sum_non_relevant_docs = 0
        for j in non_relevant_docs:
            sum_non_relevant_docs += j[1]

        term2 = [float(beta) / len(relevant_docs) * sum_relevant_docs]
        term3 = [-float(gamma) / len(non_relevant_docs) * sum_non_relevant_docs]

        # convert to a list
        term1 = [list(i) for i in term1]

        pos = 0
        while pos < len(term1):
            term1[pos][1] = term1[pos][1] + term2[0] + term3[0]
            pos += 1
        return term1

    def get_key(self, item):
        return item[1]

    def get_new_query(self, query_structure, querymod, dictionary):

        temp = querymod[:]
        temp.sort(reverse=True)
        count = 0
        for element in temp:
            pos = querymod.index(element)
            word = dictionary[pos]
            if word not in query:
                query = query + ' ' + word
                count += count
                if count == 2:
                    break

        return query_structure
