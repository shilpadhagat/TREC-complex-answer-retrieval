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

import cluster_kmeans

"""
Run this file to generate the results.run file

This file takes 7 arguments. 

outline file
paragraph file
output file
ranking function (BM25, BM25+, TFIDFIMPROVED, DIRICHLET)
number of extra clusters (int)
number of passages to extract per corpus
number of passages per assigned section

@author: Gaurav Patil.
"""

parser = argparse.ArgumentParser()
parser.add_argument("outline_file", type=str, help="Qualified location of the outline file")
parser.add_argument("paragraph_file", type=str, help="Qualified location of the paragraph file")
parser.add_argument("output_file", type=str, help="Name of the output file")
parser.add_argument("ranking_function", type=str, help="BM25, BM25+, TFIDFIMPROVED, DIRICHLET")
parser.add_argument("number of clusters", type=int, help="Number of extra clusters for kmeans step")
parser.add_argument("passages_extract",type=int, help="no of passages to extract")
parser.add_argument("passages per section", type=int, help="number of passages per section")
args = vars(parser.parse_args())

num_clusters = 0
query_cbor = args['outline_file']
paragraphs_cbor = args['paragraph_file']
output_file_name = args['output_file']
algorithm = args['ranking_function']
passages_extract = args['passages_extract']
num_clusters = args['number of clusters']
passages_per_section = args['passages per section']

query_structure = None
document_structure = None
logic_instance = None
document_texts = None
page_structure = None
queryCollections = None


if algorithm == 'BM25':
    ranking = Ranking(query_cbor, paragraphs_cbor, passages_extract)
    query_structure = ranking.gather_queries()
    document_structure = ranking.gather_paragraphs()
    logic_instance = BM25(query_structure, document_structure)
    document_texts = ranking.gather_paragraphs_plain()
    page_structure = ranking.gather_pages()

    print("No of queries" + str(len(query_structure)))
    print("No of documents" + str(len(document_structure)))

    #generate a list of sectionIds for each page
    queryCollections = list()
    for pageobj in page_structure:
        mypageid = pageobj.page_id #formatted pageid
        mypagename = pageobj.page_name #plaintext pagename
        sectionIds = list()
        sectionNames = list()
        for section in pageobj.flat_headings_list():
            sectionpath = mypageid + "/"
            sectionphrase = mypagename + " "
            for child in section:
                sectionpath += child.headingId + "/"
                sectionphrase += child.heading + " "
            sectionIds.append(sectionpath[0:-1]) #clip off the last "/"
            sectionNames.append(sectionphrase[0:-1]) #clip off the last space
        queryCollections.append((mypageid, mypagename, deepcopy(sectionNames),deepcopy(sectionIds)))
   
elif algorithm == 'BM25+':
    ranking = Ranking(query_cbor, paragraphs_cbor, passages_extract)
    query_structure = ranking.gather_queries()
    document_structure = ranking.gather_paragraphs()
    logic_instance = BM25PLUS(query_structure, document_structure)
    document_texts = ranking.gather_paragraphs_plain()
    page_structure = ranking.gather_pages()

    print("No of queries" + str(len(query_structure)))
    print("No of documents" + str(len(document_structure)))

    #generate query names for each page
    queryCollections = list()
    for pageobj in page_structure:
        mypageid = pageobj.page_id #formatted pageid
        mypagename = pageobj.page_name #plaintext pagename
        sectionIds = list()
        sectionNames = list()
        for section in pageobj.flat_headings_list():
            sectionpath = mypageid + "/"
            sectionphrase = mypagename + " "
            for child in section:
                sectionpath += child.headingId + "/"
                sectionphrase += child.heading + " "
            sectionIds.append(sectionpath[0:-1]) #clip off the last "/"
            sectionNames.append(sectionphrase[0:-1]) #clip off the last space
        queryCollections.append((mypageid, mypagename, deepcopy(sectionNames),deepcopy(sectionIds)))


elif algorithm == 'TFIDFIMPROVED':

    ranking = Ranking(query_cbor, paragraphs_cbor, passages_extract)
    query_structure = ranking.gather_queries()
    document_structure = ranking.gather_paragraphs()
    logic_instance = TDELTAIDF(query_structure, document_structure)
    document_texts = ranking.gather_paragraphs_plain()
    page_structure = ranking.gather_pages()

    print("No of queries" + str(len(query_structure)))
    print("No of documents" + str(len(document_structure)))

    #generate query names for each page
    queryCollections = list()
    for pageobj in page_structure:
        mypageid = pageobj.page_id #formatted pageid
        mypagename = pageobj.page_name #plaintext pagename
        sectionIds = list()
        sectionNames = list()
        for section in pageobj.flat_headings_list():
            sectionpath = mypageid + "/"
            sectionphrase = mypagename + " "
            for child in section:
                sectionpath += child.headingId + "/"
                sectionphrase += child.heading + " "
            sectionIds.append(sectionpath[0:-1]) #clip off the last "/"
            sectionNames.append(sectionphrase[0:-1]) #clip off the last space
        queryCollections.append((mypageid, mypagename, deepcopy(sectionNames),deepcopy(sectionIds)))

elif algorithm == 'DIRICHLET':

    ranking = Ranking(query_cbor, paragraphs_cbor, passages_extract)
    query_structure = ranking.gather_queries()
    document_structure = ranking.gather_paragraphs()
    logic_instance = DIRICHLET(query_structure, document_structure, 2500)
    document_texts = ranking.gather_paragraphs_plain()
    page_structure = ranking.gather_pages()

    print("No of queries" + str(len(query_structure)))
    print("No of documents" + str(len(document_structure)))

    #generate query names for each page
    queryCollections = list()
    for pageobj in page_structure:
        mypageid = pageobj.page_id #formatted pageid
        mypagename = pageobj.page_name #plaintext pagename
        sectionIds = list()
        sectionNames = list()
        for section in pageobj.flat_headings_list():
            sectionpath = mypageid + "/"
            sectionphrase = mypagename + " "
            for child in section:
                sectionpath += child.headingId + "/"
                sectionphrase += child.heading + " "
            sectionIds.append(sectionpath[0:-1]) #clip off the last "/"
            sectionNames.append(sectionphrase[0:-1]) #clip off the last space
        queryCollections.append((mypageid, mypagename, deepcopy(sectionNames),deepcopy(sectionIds)))

else:
    print("Invalid ranking function")
    exit()

# Generate the query scores
print("Generating the output structure by calculating scores................\n")
query_scores = dict()
queries_parsed = 0
for query in query_structure:
    temp_list = []
    print(queries_parsed)
    for key, value in document_structure.items():
        temp_list.append(logic_instance.score(query, key))
    temp_list.sort(key=lambda m: m[2])
    temp_list.reverse()
    query_scores[query[1]] = deepcopy(temp_list[0:passages_per_section])
    queries_parsed += 1

writeMode = "w" #first write of clustering output not appending

#generate the input for kmeans: (pagename(plaintext), section_names, paragraphs((id,text) list), queryids)
for collection in queryCollections:
    data = list()
    data.append(collection[1])
    data.append(collection[2])
    paragraphs = set()
    for queryid in collection[3]:
        for scoretup in query_scores[queryid]:
            paragraphs = paragraphs | {(scoretup[1],document_texts[scoretup[1]])}
    data.append(list(paragraphs))
    data.append(collection[3])
    rankings = cluster_kmeans.runKMeansPipeline(data, num_clusters)

    print("Writing one page's output to file...............................................\n")

    with open(output_file_name, mode=writeMode, encoding='UTF-8') as f:
        writer = f
        temp_list = []
        for rankingsList in rankings:
            for ranking in rankingsList:
                temp_list.append(RankingEntry(ranking[2], ranking[1], rankingsList.index(ranking) + 1, ranking[0]))
        format_run(writer, temp_list, exp_name='test')
        f.close()
    writeMode = "a" #all further writes to append to file

