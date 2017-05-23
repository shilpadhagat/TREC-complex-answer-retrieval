#!/usr/bin/python3

import argparse
import numpy
import itertools
from pprint import pprint
from copy import deepcopy

import sklearn.feature_extraction.text as skTextFeatures
import sklearn.pipeline as skPipeline
import sklearn.cluster as skCluster
import sklearn.metrics.pairwise as pairwise
from tc_Ranking import Ranking

NUM_EXTRA_CLUSTERS = 0

def mapToNames(bagsOfWords, section_names):
    maps = {}
    basicMaps = {}
    vectorizer = skTextFeatures.TfidfVectorizer()
    sectionsLength = len(bagsOfWords)

    #Put everything in one bag so we can get cosine similarities
    grabBag = deepcopy(section_names)
    
    for name in bagsOfWords:
        grabBag.append(deepcopy(name))

    grabBagVectors = vectorizer.fit_transform(grabBag)
    similarities = pairwise.cosine_similarity(grabBagVectors)

    for name in section_names:
        value = numpy.argmax(similarities[section_names.index(name)][len(section_names):])
        maps[name] = value

    return maps

def generateRanking(sectionName, bagOfParagraphs, labels):
    "Generates a list of (score, passageId, sectionName) tuples"
    ranking = {}
    workList = [sectionName] + bagOfParagraphs
    vectorizer = skTextFeatures.TfidfVectorizer()
    rankingVector = vectorizer.fit_transform(workList)
    similarities = pairwise.cosine_similarity(rankingVector)
    
    #set up hash between labels and scores
    scores = similarities[0][1:]
    for i in range(len(scores)):
        ranking[labels[i]] = scores[i]

    #generate our output list for a ranking
    sortedList = list((ranking.get(i), i, sectionName) for i in ranking.keys())
    sortedList = sorted(sortedList, key=lambda x: x[0], reverse=True)

    #debug: print out test ranking
    """
    print("SectionName: %s" %(sectionName))
    for i in range(1,len(labels)+1):
        print("%i %s %f" %(i, sortedList[i-1][1],sortedList[i-1][0]))
    print("\n\n")
    """
    return(sortedList)

def runKMeansPipeline(myData, num_clusters=None, processText=False):
    """Runs kmeans on a single page's worth of passages and section names
    Input: pagename (str), query names (section_names) (str), cluster_paragraphs (list[passageids,passagetext]),
    queryIds (string)
    """
    pageName = myData[0]
    section_names = myData[1]
    cluster_paragraphs = myData[2] #[0] is id, [1] is text
    queryids = myData[3]
    
    if(num_clusters is not None):
        NUM_EXTRA_CLUSTERS = num_clusters

    section_names_formatted = [Ranking.process_text_query_plain(section_name) for section_name in section_names]
    
    cluster_pTexts = [paragraph[1] for paragraph in cluster_paragraphs]
    numClusters = len(section_names) + NUM_EXTRA_CLUSTERS 
    print("\n\nRunning Kmeans on page ''%s''\n" %(pageName))
    
    if(len(cluster_pTexts) < numClusters):
        print("Fewer paragraphs to cluster than clusters, setting number of clusters = number of paragraphs")
        numClusters = len(cluster_pTexts)

    print("Number of clusters %i Number of paragraphs %i\n" %(numClusters, len(cluster_pTexts)))
    vectorizer = skTextFeatures.TfidfVectorizer()
    cluster_vectors = vectorizer.fit_transform(cluster_pTexts)

    km = skCluster.KMeans(n_clusters=numClusters, init='k-means++', max_iter=100, n_init=10)
    km.fit(cluster_vectors)

    finalClusters = [[] for elem in range(numClusters)]
    finalLabels = [[] for elem in range(numClusters)]

    for i in range(len(km.labels_)):
        finalClusters[km.labels_[i]].append(deepcopy(cluster_paragraphs[i][1]))
        finalLabels[km.labels_[i]].append(deepcopy(cluster_paragraphs[i][0]))

    bagsOfWords = ["" for elem in range(numClusters)]
    for i in range(numClusters):
        for j in finalClusters[i]:
            bagsOfWords[i] += j

    maps = mapToNames(bagsOfWords, section_names_formatted)
    rankings = []
    for name, queryid in zip(section_names_formatted, queryids):
        rankings.append(generateRanking(queryid,finalClusters[maps[name]],finalLabels[maps[name]]))
    return rankings