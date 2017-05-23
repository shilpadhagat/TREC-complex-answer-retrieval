import _pickle
import os
from tc_TFIDF_IMPROVED import TDELTAIDF
from copy import deepcopy
from trec_car.format_runs import *
import gc
iterator = 0

query_structure = _pickle.load(open(os.path.join(os.curdir, "cache/train_queries"), "rb"))

while iterator != 2:
    print("FileNo: " + str(iterator))
    output_file_name = "partial_files/" + "result_set" + str(iterator) + ".run"
    document_structure = _pickle.load(open(os.path.join(os.curdir, "merge_cache/para_collection"+str(iterator)), "rb"))
    logic_instance = TDELTAIDF(query_structure, document_structure)
    # Generate the query scores
    print("Generating the output structure by calculating scores................\n")
    query_scores = dict()
    queries_parsed = 0
    for query in query_structure[:10]:
        temp_list = []
        top_n_list = []
        print(queries_parsed)
        for key, value in document_structure.items():
            temp_list.append(logic_instance.score(query, key))
        temp_list.sort(key=lambda m: m[2])
        temp_list.reverse()
        for elem in temp_list[:100]:
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
    query_scores.clear()
    gc.collect()
    iterator += 1
