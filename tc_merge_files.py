"""
Note: This is a static code written only for the purposes of testing 7 million on server
For actual file generation refer to regular parameter taking files like tc_rerank_document_framework.py ( Git readme )
"""
import os
iterator_duo = 0

name_list = []
paths = os.listdir('partial_files')
for path in paths:
    name_list.append('partial_files/' + path)

with open("train.run", 'w') as outfile:
    for fname in name_list:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)

