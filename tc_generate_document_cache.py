import _pickle
import argparse
import os

from tc_Ranking import Ranking

"""
Before using the cache based implementation run this file this creates the cache required to speed up things
in the ranking functions.

The caches are stored in the ./cache directory.

"""

parser = argparse.ArgumentParser()
parser.add_argument("outline_file", type=str, help="Qualified location of the outline file")
parser.add_argument("paragraph_file", type=str, help="Qualified location of the paragraph file")
parser.add_argument("passages_extract",type=int, help="no of passages to extract")
parser.add_argument("tagme_enhanced", type=str, help="enhanced or un_enhanced")
args = vars(parser.parse_args())

query_cbor = args['outline_file']
paragraphs_cbor = args['paragraph_file']
passages_extract = args['passages_extract']
tagme_flag = args['tagme_enhanced']

ranking = Ranking(query_cbor, paragraphs_cbor, passages_extract, enable_cache=True)
query_structure = None
document_structure = None

if tagme_flag == "un_enhanced":
    query_structure = ranking.gather_queries()
    document_structure = ranking.gather_paragraphs()
elif tagme_flag == "enhanced":
    query_structure = ranking.gather_entity_enhanced_queries_mentions()
    document_structure = ranking.gather_entity_enhanced_paragraphs_mentions()
else:
    print("Select enhanced or un_enhanced")
    exit()

# Build cache for no of documents containing a specific word:
no_of_docs_with_term = dict()
for elem in query_structure:
    for key, value in elem[2].items():
        for k, v in document_structure.items():
            if key in v:
                if key in no_of_docs_with_term:
                    no_of_docs_with_term[key] += 1
                else:
                    no_of_docs_with_term[key] = 1
print(" no_of_docs_with_term cache create successfully ")
_pickle.dump(no_of_docs_with_term, open(os.path.join(os.curdir, "cache/no_of_docs_with_term"), "wb"))


# Build cache for average document length
summ = 0
for para_id, ranked_words_dict in document_structure.items():
    summ += sum(ranked_words_dict.values())
average = summ / float(len(document_structure))
print(" average_length_cache cache create successfully ")
_pickle.dump(average, open(os.path.join(os.curdir, "cache/average_length_of_documents"), "wb"))


# Build cache for DIRICHLET
# Total no of words in the collection
for elem in query_structure:
    summ += sum(elem[2].values())
print(" no of words in collection cache create successfully ")
_pickle.dump(summ, open(os.path.join(os.curdir, "cache/no_of_words_in_the_collection"), "wb"))

# Frequency of all terms in the dictionary
all_terms_frequency_dict = dict()

for kkk, vvv in document_structure.items():
    for kkkk, vvvv in vvv.items():
        if kkkk in all_terms_frequency_dict:
            all_terms_frequency_dict[kkkk] += vvvv
        else:
            all_terms_frequency_dict[kkkk] = vvvv
for ele in query_structure:
    for kkkkk, vvvvv in ele[2].items():
        if kkkkk in all_terms_frequency_dict:
            all_terms_frequency_dict[kkkkk] += vvvvv
        else:
            all_terms_frequency_dict[kkkkk] = vvvvv
print(" all_terms_freq_dict cache create successfully ")
_pickle.dump(all_terms_frequency_dict, open(os.path.join(os.curdir, "cache/all_terms_freq_dict"), "wb"))



