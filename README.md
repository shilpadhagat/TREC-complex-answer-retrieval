# complex-answer-retrieval (TREC)
This is an implementation for the TREC-CAR track found at http://trec-car.cs.unh.edu/

```
Clone this repository:

git clone https://github.com/shilpadhagat/trec-complex-answer-retrieval

```

```
List of dependencies: ( No changes from prototype 2 )

Run on Python 3.5.2
NLTK           :conda install -c anaconda nltk=3.2.2
TREC_CAR_TOOLS :https://github.com/TREMA-UNH/trec-car-tools
RE             :Regular expressions
Stemming       :https://pypi.python.org/pypi/stemming/1.0
_pickle        :Python 3.5 <built_in>
argparse       :conda install -c anaconda argparse=1.3.0
numpy          :conda install -c anaconda numpy=1.12.1 
scipy          :conda install -c anaconda scipy=0.19.0
scikit-learn   :conda install scikit-learn
TagMe          : pip install tagme (GCUBE_TOKEN = "bfbfb535-3683-47c0-bd11-df06d5d96726-843339462")
```
```
Running the code:

The code implements 4 types of ranking functions BM25, BM25+, TFIDF(Delta), DIRICHLET

Caching mechanism implemented for TFIDF(Delta) and DIRICHLET SMOOTHING METHODS so once the cache is generated it can used for both of these methods.

If running tests on TFIDF(Delta) and DIRICHLET
```
# Generating Cache ( Note: Updated in Prototype 3 )
```
** Caching now works with tagme queries ** 

tc_generate_document_cache.py [outlines file] [paragraphs file] [no of passages to extracts from paragraph file] [use_tagme_enhancement]

[use_tagme_enhancement] : enhanced, un_enhanced
```

# (top-n re-ranked)Generating trec_eval compatible results using top n re-ranking algorithm ( Note: New in prototype 3 )
```
python tc_rerank_document_framework.py [outlines file] [paragraphs file] [output file] [primary_retrieval_algorithm] [re-ranking_algorithm] [cache] [no_of_results_to_re-rank] [no of passages to extract] [use_tagme_enhancement]

The aforementoned arguments can take the following value:

[primary_retrieval_algorithm]: BM25, BM25+, TFIDFIMPROVED
[re-ranking_algorithm]       : DIRICHLET
[cache]                      : no_cache, cache ( Note 'cache' only works if tc_generate_document_cache.py is run first on same number of passages )
[no_of_results_to_re-rank]   : an integer (less than no of passages being extracted)
[no of passages to extract]  : an integer
[use_tagme_enhancement]      : enhanced, un_enhanced

Sample Run Statement:

Note: For a quick result evaluation use the un_enhanced but if you want to use enhanced generate the cache first.

python tc_rerank_document_framework.py all.test200.cbor.outlines release-v1.4.paragraphs output.top500reranked.run TFIDFIMPROVED DIRICHLET no_cache 500 50000 un_enhanced
```

# (top-n not re-ranked)Generating trec_eval compatible results top - n without reranking ( Note: New in prototype 3 )
```
**** Note:
This is only to compare the top n implementation as results from top n might be a little bit lower than the the entire thing this is a good way to generate a top n only results file without re ranking to compare it with the re-ranked implementation
***** 

python tc_rerank_document_framework.py [outlines file] [paragraphs file] [output file] [ranking_function] [cache] [top_n_only] [no of passages to extract] [use_tagme_enhancement]

The aforementoned arguments can take the following value:

[ranking_function]: BM25, BM25+, TFIDFIMPROVED, DIRICHLET
[cache]                      : no_cache, cache ( Note 'cache' only works if tc_generate_document_cache.py is run first on same number of passages )
[top_n_results_only]         : an integer (less than no of passages being extracted)
[no of passages to extract]  : an integer
[use_tagme_enhancement]      : enhanced, un_enhanced

Sample Run Statement:

Note: For a quick result evaluation use the un_enhanced but if you want to use enhanced generate the cache first.

python tc_rerank_document_framework.py all.test200.cbor.outlines release-v1.4.paragraphs output.top500notreranked.run DIRICHLET no_cache 500 50000 un_enhanced
```



# Generating trec_eval compatible results file all results ( Note: Updated in Prototype 3 )
```
tc_generate_document.py [outlines file] [paragraphs file] [output file] [ranking function] [cache] [no of passages to extract] [use_tagme_enhancement]

The aforementoned arguments can take the following value:

[ranking function]          : BM25, BM25+, TFIDFIMPROVED, DIRICHLET
[cache]                     : no_cache, cache ( Note 'cache' only works if tc_generate_document_cache.py is run first on same number of passages )
[no of passages to extract] : an integer
[use_tagme_enhancement]     : enhanced, un_enhanced
```


```
Running the code with or without entity linking:

This script runs using main approach with entity linking on test200 data

chmod +x entitylinking.run.sh
./entitylinking.run.sh

This script runs using baseline approach (BM25) on test200 data

chmod +x baseline.run.sh
./baseline.run.sh

```

# NOTE
```
Important note about caching. cPickle has limitations to the amount of data that it can serizalize and store so after running a test on 7 million passages trying to cache them proved that indexes of such a large data set cannot be cached

For the next iteration a pure pythonic full text indexing library can be used such as 
https://pypi.python.org/pypi/Whoosh/
or an alternative to this would be to serialize and store the data in a relational or no-sql database.
```
# Calculating Results

```
eval framework from trec_car 

eval_framework.py [qrels] [run]
```
# Clustering Implementation
```
Implemented by Colin
in folder trec_cluster_basic run the run.sh script to get a sample clustering followed by ranking
Accepts inputs and explains possible input values on running with no arguments.


Now (prototype 3) synced up with guarav's implementation in trec_cluster_generate_document.py, but no caching achieved

to run 

python3 trec_cluster_generate_document.py [outline.cbor] [paragraphs.cbor] [outputfilename] [rankingfunction] [number of additional clusters] [number of passages to extract] [passages per section]

Clustering on .run files achieved with tc_clusterRunFile

to run:

python3 tc_clusterRunFile.py [input runfile name] [outline.cbor] [paragraphs.cbor] [output runfile name] [number of additional clusters] [max passages per section]
```
# Entity Linking with relevance model Implementation
```
First create Cache for Tagme me enhanced data using Gaurav's Cache implemenatation:
tc_generate_document_cache.py 
This file takes three input parameters:
Outline file, paragraph file and number of passages you want to extract.
Example run: 
python tc_generate_document_cache.py all.test200.cbor.outlines release-v1.4.paragraphs 50000

tc_generate_entitylink_rm_cache_results.py for performing entity linking with query expansion using rocchio relevance model. This will generate output file used in evaluation framework. This file takes 6 parameters:
outlines file, paragraph file, output file, retrieval algorithm, use_cache and number of passages to extract
Example run: 
python tc_generate_entitylink_rm_cache_results.py all.test200.cbor.outlines release-v1.4.paragraphs output.run BM25 cache 50000

Finally run the evaluation framework:
eval_framework.py This is for evaluating the results. This takes two parameters:
qrels file and output file
Example run: 
	python eval_framework.py all.test200.cbor.hierarchical.qrels output.run
```

# Results (Updated from Prototype 3)

```
**************** Prototype 3 Results: ****************
============================================================================================================================
Gaurav's Results
============================================================================================================================
using test 200's hierarchical qrel file and 7,000,000 passages from release1.4.v

Scoring with TFIDF improved and reranking top 100 with Dirichlet ( Less penalty for lengthly documents )
map   : 0.139   
r-prec: 0.140

Scoring with TFIDF improved and reranking top 1000 with Dirichlet ( Less penalty for lengthly documents )
map   : 0.141   
r-prec: 0.145

Scoring with prototype 2's Dirichlet no reranking top 1000 
map   : 0.157
r-prec: 0.137

using test 200's hierarchical qrel file and 1,000,000 passages from release1.4.v
Primary Scoring Algorithm: TFIDF Improved ( was better compared with BM25, BM25+ )
Re-ranking Algorithm:      DIRICHLET

Reranking top 10000:
map   :0.15
r-prec:0.21

top 10000 without re-ranking:

map   :0.13
r-prec:0.15

Experiment for comparing our pipelines:
50,000 passages release1.4.v test200's hierarchical qrel.

map    :0.0031
r-prec :0.0027

*** 
finally was able to create index for all 7million passages. You can find them in merge cache.
Then made partial run files and then ran the evaluation as a mega file.
tc_test_7million.py
tc_modified_ranking_7million.py
tc_merge_files.py



============================================================================================================================
Shilpa's Results
============================================================================================================================
Test200 outline and qrel file and 7,000,000 passages from release1.4.v
Entity link with Dirichlet 

map	: 0.1377
r-prec  : 0.1332

Experiment for comparing our pipelines:
50,000 passages release1.4.v test200's hierarchical qrel.

mrr     :0.0012
p@5     :0.00036
r-prec  :0.00075
map     :0.00082

============================================================================================================================
Colin's Results
============================================================================================================================

using test200's hierarchical qrel file and all 7000000 passages from release1.4
input runfile generated by Guarav's TFIDF Improved and reranking top 100 with Dirichlet

python3 tc_clusterRunFile.py 7millionTFIDFIMPROVED.run all.
test200.cbor.outlines release-v1.4.paragraphs 7MILTFIDFcluster2_10.run 2 10

map                   	all	0.0009
Rprec                 	all	0.0009


using test 200's hierarchichal qrel file and 50,000 passages from release1.4.

50,000 passages release1.4.v test200's hierarchical qrel.
python3 trec_cluster_generate_document.py all.test200.cbor.outlines release-v1.4.paragraphs TFIDFIMPROVED_cluster.run TFIDFIMPROVED 6 50000 20 

map                     all 0.0009
Rprec                   all 0.0009

**************** Prototype 2 Results: ****************
DIRICHLETS SMOOTHING ALGORITHM

using test 200's hierarchichal qrel file and 1,000,000 passages from release1.4.v

map - 0.1423
r-prec - 0.1311

using test 200's hierarchichal qrel file and 4000 passages from training data

map - 0.2782
r-prec - 0.2303

Entity linking performance using test200 data with 4000 passages:

mrr = 0.341,
p@5=0.126,
r-prec=0.191,
map=0.230

Entity linking performance using test200 data with 4000 passages:

mrr=0.30,
p@5=0.11,
r-prec=0.182,
map=0.223)


```

# Contributions Prototype 3
```
Gaurav Patil: 
Re-ranking top n queries using multiple retrieval algorithms,
Retrieving top n results for comparision, 
Expanded cache to all retrieval algorithms implemented in prototype 2, 
Implemented caching for tagme enhanced queries and passages, 
Updated data structures for existing retrieval methods, 
Merged entity linking from prototype 2 to re-ranking module to complete the pipeline, 
Refactored redundant code and removed minimized non-essential code.
Created test suite and debugged the problem from last prototype. 

Shilpa Dhagat:
Implemented Rocchio algorithm based on Relevance feedback,
Merged entity linking implementation with Gaurav's code, Code now takes a parameter to run the pipeline with or without tagme expansion.
Synced Colin's clustering with Tagme implementation.
Used Gaurav's caching and re-rank methods to use the top-100 paragraphs for entity-linking.
```

# Contributions Prototype 2
```
Gaurav Patil: BM25, BM25+, DIRICHLETS, TFIDF Improved, Text Processing Algorithms and Cachcing 
Colin Etzel : Implemented clustering and work towards integrating with DIRICHLETS (see section "Clustering Implementation")
Shilpa Dhagat: Entity linking implementation with Tagme using both annotations and metions(spots) for queries and paragraphs and Greedy interpretation finding algorithm 
```
