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



parser = argparse.ArgumentParser()
parser.add_argument("outline_file", type=str, help="Qualified location of the outline file")
parser.add_argument("paragraph_file", type=str, help="Qualified location of the paragraph file")
parser.add_argument("output_file", type=str, help="Name of the output file")
parser.add_argument("ranking_function", type=str, help="BM25, BM25+, TFIDFIMPROVED")
parser.add_argument("use_cache", type=str, help="cache, no_cache")
parser.add_argument("only_top_n", type=int, help="Select top_n results")
parser.add_argument("passages_extract", type=int, help="no of passages to extract")
parser.add_argument("tagme_enchanced", type=str, help="enhanced or un_enhanced")
args = vars(parser.parse_args())

query_cbor = args['outline_file']
paragraphs_cbor = args['paragraph_file']
output_file_name = args['output_file']
algorithm = args['ranking_function']
cache_flag = args['use_cache']
top_n = args["only_top_n"]
passages_extract = args['passages_extract']
tagme_enabled = args["tagme_enchanced"]

if passages_extract < top_n:
    print("The no of passages extracted should be greater than the number of results to be re-ranked")
    exit()

query_structure = None
document_structure = None
logic_instance = None


if tagme_enabled == "un_enhanced":
    if algorithm == 'BM25':
        if cache_flag == 'cache':
            BM25.useCache = True
            query_structure = _pickle.load(open(os.path.join(os.curdir, "cache/query_structure_cache"), "rb"))
            document_structure = _pickle.load(open(os.path.join(os.curdir, "cache/paragraph_structure"), "rb"))
            print("No of queries" + str(len(query_structure)))
            print("No of documents" + str(len(document_structure)))
            BM25.number_of_words_in_the_collection_s = \
                _pickle.load(open(os.path.join(os.curdir, "cache/no_of_words_in_the_collection"), "rb"))
            BM25.all_words_freq_dict = _pickle.load(open(os.path.join(os.curdir, "cache/all_terms_freq_dict"), "rb"))
            logic_instance = BM25(query_structure, document_structure)
        else:
            ranking = Ranking(query_cbor, paragraphs_cbor, passages_extract)
            query_structure = ranking.gather_queries()
            document_structure = ranking.gather_paragraphs()
            print("No of queries" + str(len(query_structure)))
            print("No of documents" + str(len(document_structure)))
            logic_instance = BM25(query_structure, document_structure)


    elif algorithm == 'BM25+':
        if cache_flag == 'cache':
            BM25PLUS.useCache = True
            query_structure = _pickle.load(open(os.path.join(os.curdir, "cache/query_structure_cache"), "rb"))
            document_structure = _pickle.load(open(os.path.join(os.curdir, "cache/paragraph_structure"), "rb"))
            print("No of queries" + str(len(query_structure)))
            print("No of documents" + str(len(document_structure)))
            BM25PLUS.number_of_words_in_the_collection_s = \
                _pickle.load(open(os.path.join(os.curdir, "cache/no_of_words_in_the_collection"), "rb"))
            BM25PLUS.all_words_freq_dict = _pickle.load(
                open(os.path.join(os.curdir, "cache/all_terms_freq_dict"), "rb"))
            logic_instance = DIRICHLET(query_structure, document_structure, 2500)
        else:
            ranking = Ranking(query_cbor, paragraphs_cbor, passages_extract)
            query_structure = ranking.gather_queries()
            document_structure = ranking.gather_paragraphs()
            print("No of queries" + str(len(query_structure)))
            print("No of documents" + str(len(document_structure)))
            logic_instance = BM25PLUS(query_structure, document_structure)

    elif algorithm == 'TFIDFIMPROVED':
        logic_instance = None
        if cache_flag == 'cache':
            TDELTAIDF.useCache = True
            query_structure = _pickle.load(open(os.path.join(os.curdir, "cache/query_structure_cache"), "rb"))
            document_structure = _pickle.load(open(os.path.join(os.curdir, "cache/paragraph_structure"), "rb"))
            print("No of queries" + str(len(query_structure)))
            print("No of documents" + str(len(document_structure)))
            TDELTAIDF.average_doc_length = _pickle.load(
                open(os.path.join(os.curdir, "cache/average_length_of_documents"), "rb"))
            TDELTAIDF.no_of_docs_dict = _pickle.load(open(os.path.join(os.curdir, "cache/no_of_docs_with_term"), "rb"))
            logic_instance = TDELTAIDF(query_structure, document_structure)
        else:
            ranking = Ranking(query_cbor, paragraphs_cbor, passages_extract)
            query_structure = ranking.gather_queries()
            document_structure = ranking.gather_paragraphs()
            print("No of queries" + str(len(query_structure)))
            print("No of documents" + str(len(document_structure)))
            logic_instance = TDELTAIDF(query_structure, document_structure)

elif tagme_enabled == "enhanced":
    if algorithm == 'BM25':
        if cache_flag == 'cache':
            BM25.useCache = True
            query_structure = _pickle.load(open(os.path.join(os.curdir, "cache/query_structure_cache"), "rb"))
            document_structure = _pickle.load(open(os.path.join(os.curdir, "cache/paragraph_structure"), "rb"))
            print("No of queries" + str(len(query_structure)))
            print("No of documents" + str(len(document_structure)))
            BM25.number_of_words_in_the_collection_s = \
                _pickle.load(open(os.path.join(os.curdir, "cache/no_of_words_in_the_collection"), "rb"))
            BM25.all_words_freq_dict = _pickle.load(open(os.path.join(os.curdir, "cache/all_terms_freq_dict"), "rb"))
            logic_instance = BM25(query_structure, document_structure)
        else:
            ranking = Ranking(query_cbor, paragraphs_cbor, passages_extract)
            query_structure = ranking.gather_entity_enhanced_queries_mentions()
            document_structure = ranking.gather_entity_enhanced_paragraphs_mentions()
            print("No of queries" + str(len(query_structure)))
            print("No of documents" + str(len(document_structure)))
            logic_instance = BM25(query_structure, document_structure)

    elif algorithm == 'BM25+':
        if cache_flag == 'cache':
            BM25PLUS.useCache = True
            query_structure = _pickle.load(open(os.path.join(os.curdir, "cache/query_structure_cache"), "rb"))
            document_structure = _pickle.load(open(os.path.join(os.curdir, "cache/paragraph_structure"), "rb"))
            print("No of queries" + str(len(query_structure)))
            print("No of documents" + str(len(document_structure)))
            BM25PLUS.number_of_words_in_the_collection_s = \
                _pickle.load(open(os.path.join(os.curdir, "cache/no_of_words_in_the_collection"), "rb"))
            BM25PLUS.all_words_freq_dict = _pickle.load(
                open(os.path.join(os.curdir, "cache/all_terms_freq_dict"), "rb"))
            logic_instance = DIRICHLET(query_structure, document_structure, 2500)
        else:
            ranking = Ranking(query_cbor, paragraphs_cbor, passages_extract)
            query_structure = ranking.gather_entity_enhanced_queries_mentions()
            document_structure = ranking.gather_entity_enhanced_paragraphs_mentions()
            print("No of queries" + str(len(query_structure)))
            print("No of documents" + str(len(document_structure)))
            logic_instance = BM25PLUS(query_structure, document_structure)

    elif algorithm == 'TFIDFIMPROVED':
        logic_instance = None
        if cache_flag == 'cache':
            TDELTAIDF.useCache = True
            query_structure = _pickle.load(open(os.path.join(os.curdir, "cache/query_structure_cache"), "rb"))
            document_structure = _pickle.load(open(os.path.join(os.curdir, "cache/paragraph_structure"), "rb"))
            print("No of queries" + str(len(query_structure)))
            print("No of documents" + str(len(document_structure)))
            TDELTAIDF.average_doc_length = _pickle.load(
                open(os.path.join(os.curdir, "cache/average_length_of_documents"), "rb"))
            TDELTAIDF.no_of_docs_dict = _pickle.load(open(os.path.join(os.curdir, "cache/no_of_docs_with_term"), "rb"))
            logic_instance = TDELTAIDF(query_structure, document_structure)
        else:
            ranking = Ranking(query_cbor, paragraphs_cbor, passages_extract)
            query_structure = ranking.gather_entity_enhanced_queries_mentions()
            document_structure = ranking.gather_entity_enhanced_paragraphs_mentions()
            print("No of queries" + str(len(query_structure)))
            print("No of documents" + str(len(document_structure)))
            logic_instance = TDELTAIDF(query_structure, document_structure)


# Generate the query scores
print("Generating the output structure by calculating scores................\n")
query_scores = dict()
queries_parsed = 0
for query in query_structure:
    temp_list = []
    top_n_list = []
    print(queries_parsed)
    for key, value in document_structure.items():
        temp_list.append(logic_instance.score(query, key))
    temp_list.sort(key=lambda m: m[2])
    temp_list.reverse()
    for elem in temp_list[:top_n]:
        top_n_list.append((elem[0][1], elem[1], elem[2]))
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
