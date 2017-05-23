import _pickle
import argparse
import os
from copy import deepcopy

from trec_car.format_runs import *

from tc_BM25PLUS_ranking import BM25PLUS
from tc_BM25_ranking import BM25
from tc_DIRICHLET import DIRICHLET
from tc_Ranking import Ranking
from tc_TFIDF_IMPROVED import TDELTAIDF

"""
This program is the implementation of the re-ranking the top n retrieved.
"""

parser = argparse.ArgumentParser()
parser.add_argument("outline_file", type=str, help="Qualified location of the outline file")
parser.add_argument("paragraph_file", type=str, help="Qualified location of the paragraph file")
parser.add_argument("output_file", type=str, help="Name of the output file")
parser.add_argument("primary_retrieval_algorithm", type=str, help="BM25, BM25+, TFIDFIMPROVED, DIRICHLET")
parser.add_argument("re-ranking_algorithm", type=str, help="BM25, BM25+, TFIDFIMPROVED, DIRICHLET")
parser.add_argument("use_cache", type=str, help="cache, no_cache")
parser.add_argument("no_of_results_to_re-rank", type=int, help="Select how many to re-rank")
parser.add_argument("passages_extract", type=int, help="no of passages to extract")
parser.add_argument("tagme_enchanced", type=str, help="enhanced or un_enhanced")
args = vars(parser.parse_args())

query_cbor = args['outline_file']
paragraphs_cbor = args['paragraph_file']
output_file_name = args['output_file']
cache_flag = args['use_cache']
primary_retrieval_algorithm = args["primary_retrieval_algorithm"]
reranking_algorithm = args["re-ranking_algorithm"]
rerank_n = args["no_of_results_to_re-rank"]
passages_extract = args["passages_extract"]
tagme_enabled = args["tagme_enchanced"]

if passages_extract < rerank_n:
    print("The no of passages extracted should be greater than the number of results to be re-ranked")
    exit()

query_structure = None
document_structure = None
primary = None
re_rank = None

if tagme_enabled == "un_enhanced":
    #
    #
    #
    # If the algorithms are BM25 and DIRICHLET
    #
    #
    #
    if primary_retrieval_algorithm == "BM25" and reranking_algorithm == "DIRICHLET":
        if cache_flag == 'cache':
            BM25.useCache = True
            query_structure = _pickle.load(open(os.path.join(os.curdir, "cache/query_structure_cache"), "rb"))
            document_structure = _pickle.load(open(os.path.join(os.curdir, "cache/paragraph_structure"), "rb"))
            print("No of queries" + str(len(query_structure)))
            print("No of documents" + str(len(document_structure)))
            BM25.average_doc_length = _pickle.load(
                open(os.path.join(os.curdir, "cache/average_length_of_documents"), "rb"))
            BM25.no_of_docs_dict = _pickle.load(open(os.path.join(os.curdir, "cache/no_of_docs_with_term"), "rb"))
            primary = BM25(query_structure, document_structure)

            DIRICHLET.useCache = True
            DIRICHLET.number_of_words_in_the_collection_s = \
                _pickle.load(open(os.path.join(os.curdir, "cache/no_of_words_in_the_collection"), "rb"))
            DIRICHLET.all_words_freq_dict = _pickle.load(
                open(os.path.join(os.curdir, "cache/all_terms_freq_dict"), "rb"))
            re_rank = DIRICHLET(query_structure, document_structure, 2500)
        else:
            ranking = Ranking(query_cbor, paragraphs_cbor, passages_extract)
            query_structure = ranking.gather_queries()
            document_structure = ranking.gather_paragraphs()
            primary = BM25(query_structure, document_structure)
            re_rank = DIRICHLET(query_structure, document_structure, 2500)
    #
    #
    #
    # If the algorithms are BM25+ and DIRICHLET
    #
    #
    #
    elif primary_retrieval_algorithm == "BM25+" and reranking_algorithm == "DIRICHLET":
        if cache_flag == 'cache':
            BM25PLUS.useCache = True
            query_structure = _pickle.load(open(os.path.join(os.curdir, "cache/query_structure_cache"), "rb"))
            document_structure = _pickle.load(open(os.path.join(os.curdir, "cache/paragraph_structure"), "rb"))
            print("No of queries" + str(len(query_structure)))
            print("No of documents" + str(len(document_structure)))
            BM25PLUS.average_doc_length = _pickle.load(
                open(os.path.join(os.curdir, "cache/average_length_of_documents"), "rb"))
            BM25PLUS.no_of_docs_dict = _pickle.load(open(os.path.join(os.curdir, "cache/no_of_docs_with_term"), "rb"))
            primary = BM25PLUS(query_structure, document_structure)

            DIRICHLET.useCache = True
            DIRICHLET.number_of_words_in_the_collection_s = \
                _pickle.load(open(os.path.join(os.curdir, "cache/no_of_words_in_the_collection"), "rb"))
            DIRICHLET.all_words_freq_dict = _pickle.load(
                open(os.path.join(os.curdir, "cache/all_terms_freq_dict"), "rb"))
            re_rank = DIRICHLET(query_structure, document_structure, 2500)
        else:
            ranking = Ranking(query_cbor, paragraphs_cbor, passages_extract)
            query_structure = ranking.gather_queries()
            document_structure = ranking.gather_paragraphs()
            primary = BM25PLUS(query_structure, document_structure)
            re_rank = DIRICHLET(query_structure, document_structure, 2500)
    #
    #
    #
    # If the algorithms are TFIDFIMPORVED and DIRICHLET
    #
    #
    #
    elif primary_retrieval_algorithm == "TFIDFIMPROVED" and reranking_algorithm == "DIRICHLET":
        if cache_flag == 'cache':
            TDELTAIDF.useCache = True
            query_structure = _pickle.load(open(os.path.join(os.curdir, "cache/query_structure_cache"), "rb"))
            document_structure = _pickle.load(open(os.path.join(os.curdir, "cache/paragraph_structure"), "rb"))
            print("No of queries" + str(len(query_structure)))
            print("No of documents" + str(len(document_structure)))
            TDELTAIDF.average_doc_length = _pickle.load(
                open(os.path.join(os.curdir, "cache/average_length_of_documents"), "rb"))
            TDELTAIDF.no_of_docs_dict = _pickle.load(open(os.path.join(os.curdir, "cache/no_of_docs_with_term"), "rb"))
            primary = TDELTAIDF(query_structure, document_structure)

            DIRICHLET.useCache = True
            DIRICHLET.number_of_words_in_the_collection_s = \
                _pickle.load(open(os.path.join(os.curdir, "cache/no_of_words_in_the_collection"), "rb"))
            DIRICHLET.all_words_freq_dict = _pickle.load(
                open(os.path.join(os.curdir, "cache/all_terms_freq_dict"), "rb"))
            re_rank = DIRICHLET(query_structure, document_structure, 2500)
        else:
            ranking = Ranking(query_cbor, paragraphs_cbor, passages_extract)
            query_structure = ranking.gather_queries()
            document_structure = ranking.gather_paragraphs()
            primary = TDELTAIDF(query_structure, document_structure)
            re_rank = DIRICHLET(query_structure, document_structure, 2500)

elif tagme_enabled == "enhanced":
    #
    #
    #
    # If the algorithms are BM25 and DIRICHLET
    #
    #
    #
    if primary_retrieval_algorithm == "BM25" and reranking_algorithm == "DIRICHLET":
        if cache_flag == 'cache':
            BM25.useCache = True
            query_structure = _pickle.load(open(os.path.join(os.curdir, "cache/query_structure_cache"), "rb"))
            document_structure = _pickle.load(open(os.path.join(os.curdir, "cache/paragraph_structure"), "rb"))
            print("No of queries" + str(len(query_structure)))
            print("No of documents" + str(len(document_structure)))
            BM25.average_doc_length = _pickle.load(
                open(os.path.join(os.curdir, "cache/average_length_of_documents"), "rb"))
            BM25.no_of_docs_dict = _pickle.load(open(os.path.join(os.curdir, "cache/no_of_docs_with_term"), "rb"))
            primary = BM25(query_structure, document_structure)

            DIRICHLET.useCache = True
            DIRICHLET.number_of_words_in_the_collection_s = \
                _pickle.load(open(os.path.join(os.curdir, "cache/no_of_words_in_the_collection"), "rb"))
            DIRICHLET.all_words_freq_dict = _pickle.load(
                open(os.path.join(os.curdir, "cache/all_terms_freq_dict"), "rb"))
            re_rank = DIRICHLET(query_structure, document_structure, 2500)
        else:
            ranking = Ranking(query_cbor, paragraphs_cbor, passages_extract)
            query_structure = ranking.gather_entity_enhanced_queries_mentions()
            document_structure = ranking.gather_entity_enhanced_paragraphs_mentions()
            primary = BM25(query_structure, document_structure)
            re_rank = DIRICHLET(query_structure, document_structure, 2500)
    #
    #
    #
    # If the algorithms are BM25+ and DIRICHLET
    #
    #
    #
    elif primary_retrieval_algorithm == "BM25+" and reranking_algorithm == "DIRICHLET":
        if cache_flag == 'cache':
            BM25PLUS.useCache = True
            query_structure = _pickle.load(open(os.path.join(os.curdir, "cache/query_structure_cache"), "rb"))
            document_structure = _pickle.load(open(os.path.join(os.curdir, "cache/paragraph_structure"), "rb"))
            print("No of queries" + str(len(query_structure)))
            print("No of documents" + str(len(document_structure)))
            BM25PLUS.average_doc_length = _pickle.load(
                open(os.path.join(os.curdir, "cache/average_length_of_documents"), "rb"))
            BM25PLUS.no_of_docs_dict = _pickle.load(open(os.path.join(os.curdir, "cache/no_of_docs_with_term"), "rb"))
            primary = BM25PLUS(query_structure, document_structure)

            DIRICHLET.useCache = True
            DIRICHLET.number_of_words_in_the_collection_s = \
                _pickle.load(open(os.path.join(os.curdir, "cache/no_of_words_in_the_collection"), "rb"))
            DIRICHLET.all_words_freq_dict = _pickle.load(
                open(os.path.join(os.curdir, "cache/all_terms_freq_dict"), "rb"))
            re_rank = DIRICHLET(query_structure, document_structure, 2500)
        else:
            ranking = Ranking(query_cbor, paragraphs_cbor, passages_extract)
            query_structure = ranking.gather_entity_enhanced_queries_mentions()
            document_structure = ranking.gather_entity_enhanced_paragraphs_mentions()
            primary = BM25PLUS(query_structure, document_structure)
            re_rank = DIRICHLET(query_structure, document_structure, 2500)
    #
    #
    #
    # If the algorithms are TFIDFIMPORVED and DIRICHLET
    #
    #
    #
    elif primary_retrieval_algorithm == "TFIDFIMPROVED" and reranking_algorithm == "DIRICHLET":
        if cache_flag == 'cache':
            TDELTAIDF.useCache = True
            query_structure = _pickle.load(open(os.path.join(os.curdir, "cache/query_structure_cache"), "rb"))
            document_structure = _pickle.load(open(os.path.join(os.curdir, "cache/paragraph_structure"), "rb"))
            print("No of queries" + str(len(query_structure)))
            print("No of documents" + str(len(document_structure)))
            TDELTAIDF.average_doc_length = _pickle.load(
                open(os.path.join(os.curdir, "cache/average_length_of_documents"), "rb"))
            TDELTAIDF.no_of_docs_dict = _pickle.load(open(os.path.join(os.curdir, "cache/no_of_docs_with_term"), "rb"))
            primary = TDELTAIDF(query_structure, document_structure)

            DIRICHLET.useCache = True
            DIRICHLET.number_of_words_in_the_collection_s = \
                _pickle.load(open(os.path.join(os.curdir, "cache/no_of_words_in_the_collection"), "rb"))
            DIRICHLET.all_words_freq_dict = _pickle.load(
                open(os.path.join(os.curdir, "cache/all_terms_freq_dict"), "rb"))
            re_rank = DIRICHLET(query_structure, document_structure, 2500)
        else:
            ranking = Ranking(query_cbor, paragraphs_cbor, passages_extract)
            query_structure = ranking.gather_entity_enhanced_queries_mentions()
            document_structure = ranking.gather_entity_enhanced_paragraphs_mentions()
            primary = TDELTAIDF(query_structure, document_structure)
            re_rank = DIRICHLET(query_structure, document_structure, 2500)
else:
    print("Use enhanced or un_enhanced")
    exit()

# Generate the query scores
print("Generating the output structure by calculating scores................\n")
query_scores = dict()
queries_parsed = 0
for query in query_structure:
    temp_list = []
    top_n_list = []
    print(queries_parsed)
    for key, value in document_structure.items():
        temp_list.append(primary.score(query, key))
    temp_list.sort(key=lambda m: m[2])
    temp_list.reverse()
    for elements in temp_list[:rerank_n]:
        top_n_list.append(re_rank.score(elements[0], elements[1]))
    top_n_list.sort(key=lambda m: m[2])
    top_n_list.reverse()
    query_scores[query[1]] = deepcopy(top_n_list)
    queries_parsed += 1

# Write the results to a file
print("Writing output to file...............................................\n")
with open(output_file_name, mode='w', encoding='UTF-8') as f:
    writer = f
    temp_list = []
    count = 0
    for k3, value in query_scores.items():
        count += 1
        rank = 0
        for x in value:
            rank += 1
            temp_list.append(RankingEntry(x[0], x[1], rank, x[2]))
    format_run(writer, temp_list, exp_name='test')
    f.close()
