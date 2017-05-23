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
@author: Colin Etzel


"""
#command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("run file", type=str, help="Qualified location of previous .run file")
parser.add_argument("outline_file", type=str, help="Qualified location of the outline file")
parser.add_argument("paragraph_file", type=str, help="Qualified location of the paragraph file")
parser.add_argument("output_file", type=str, help="Name of .run file to create.")
parser.add_argument("number of clusters", type=int, help="Number of extra clusters for kmeans step")
parser.add_argument("passages per section", type=int, help="number of passages per section")



def readRunFile(filepath):
    "Read the .run file to be reranked with clustering."
    runFile = open(filepath, "r")
    results = dict()
    passageIDs = list()
    previousID = None
    for line in runFile.readlines():
        linecomponents = line.split(" ")
        sectionID = linecomponents[0]
        passageID = linecomponents[2]

        if(sectionID != previousID and passageIDs != None): #new section
            results[previousID] = deepcopy(passageIDs)
            passageIDs = list()
            passageIDs.append(passageID)
        else:
            passageIDs.append(passageID)
        previousID = sectionID
    return results

def makeParagraphTupleSet(passageIDs, passageDictionary,passages_per_section):
    "Create list of (paragraphid, paragraphtext) tuples given a list of passageIDs"
    passageIDs = passageIDs[:passages_per_section]
    results = list()
    for pid in passageIDs:
        results.append((pid,Ranking.process_text_query_plain(passageDictionary[pid])))
    return set(results)


# parse command line arguments
args = vars(parser.parse_args())
filepath = args["run file"]
query_cbor = args['outline_file']
paragraphs_cbor = args['paragraph_file']
output_file_name = args['output_file']
passages_extract = 7000000 #all passages
num_clusters = args['number of clusters']
passages_per_section = args['passages per section']

print("Loading data from CBOR...")
ranking = Ranking(query_cbor, paragraphs_cbor, passages_extract)
#query_structure = ranking.gather_queries()
document_texts = ranking.gather_paragraphs_plain_noprocessing()
page_structure = ranking.gather_pages()

print("Loaded structures")

#print(query_structure[0])
#generate a list of sectionIds for each page
print("Generate queryCollections...")
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
        print(sectionpath)
        sectionIds.append(sectionpath[0:-1]) #clip off the last "/"
        sectionNames.append(sectionphrase[0:-1]) #clip off the last space
    queryCollections.append((mypageid, mypagename, deepcopy(sectionNames),deepcopy(sectionIds)))



writeMode = "w" #first write of clustering output not appending
print("Reading run file...")
runResults = readRunFile(filepath)

print("\n\n")
for key in runResults.keys():
    print(key)

#generate the input for kmeans: (pagename(plaintext), section_names, paragraphs((id,text) list), queryids)
for collection in queryCollections:
    data = list()
    data.append(collection[1])
    data.append(collection[2])

    paragraphs = set()
    for queryid in collection[3]:
        try: #in case nothing assigned to queryID
            passageIDs = runResults[queryid]
            paragraphs = paragraphs | makeParagraphTupleSet(passageIDs,document_texts,passages_per_section)
        except:
            pass
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