import _pickle
import os
import re
import tagme
from nltk.corpus import stopwords
from stemming.porter2 import stem
from trec_car.read_data import *

GCUBE_TOKEN = "bfbfb535-3683-47c0-bd11-df06d5d96726-843339462"
DEFAULT_LANG = "en"
DEFAULT_TAG_API = "https://tagme.d4science.org/tagme/tag"
DEFAULT_SPOT_API = "https://tagme.d4science.org/tagme/spot"
DEFAULT_REL_API = "https://tagme.d4science.org/tagme/rel"

"""
This class contains methods that generate the structure that is used by the ranking functions.
@author: Gaurav Patil.
"""

class Ranking:

    stop_words = stopwords.words('english')
    cache_words = dict()

    def __init__(self, outline_file, paragraph_file, no_passages_to_extract, enable_cache=False):
        """
        Constructor
        :param outline_file: path of the outline file
        :param paragraph_file: path of the paragraph file
        """
        self.passages_extract = no_passages_to_extract
        self.outline_file = outline_file
        self.paragraph_file = paragraph_file
        self.enable_cache = enable_cache
        self.pages = self.gather_pages()

    def gather_pages(self):
        """
        Gets the pages from cbor
        :return: list of pages
        """
        with open(self.outline_file, 'rb') as f:
            pages = [p for p in itertools.islice(iter_annotations(f), 0, 1000)]
        return pages

    def gather_paragraphs(self):
        """
        Gets the paragraphs from the cbor
        :return: dict of dicts ( id maps to ranked dict of words ) 
        """
        id_to_text_dict = dict()
        with open(self.paragraph_file, 'rb') as f:
            for p in itertools.islice(iter_paragraphs(f), 0, self.passages_extract):
                id_to_text_dict[p.para_id] = Ranking.process_text_query(p.get_text())
        if self.enable_cache is True:
            print(" Paragraph cache created successfully ")
            _pickle.dump(id_to_text_dict, open(os.path.join(os.curdir, "cache/paragraph_structure"), "wb"))
        return id_to_text_dict

    def gather_paragraphs_plain(self):
        """
        Uses process_text_query_plain instead 
        """
        id_to_text_dict = dict()
        with open(self.paragraph_file, 'rb') as f:
            for p in itertools.islice(iter_paragraphs(f), 0, self.passages_extract):
                id_to_text_dict[p.para_id] = Ranking.process_text_query_plain(p.get_text())
        return id_to_text_dict

    def gather_paragraphs_plain_noprocessing(self):
        id_to_text_dict = dict()
        with open(self.paragraph_file, 'rb') as f:
            for p in itertools.islice(iter_paragraphs(f), 0, self.passages_extract):
                id_to_text_dict[p.para_id] = p.get_text()
        return id_to_text_dict
        
    @staticmethod
    def process_text_query(input_text: str):
        """
        Runs text processing on a given text
        :param input_text: String
        :return: ranked dictionary of processed words
        """
        # Convert characters to lower case
        input_text_to_lower = input_text.lower()
        # Remove special characters from the string
        input_text_to_lower = re.sub('[^a-zA-Z0-9 \n]', '', input_text_to_lower)
        # Remove common words using list of stop words
        filtered_words_list = [word for word in input_text_to_lower.split() if word not in Ranking.stop_words]
        # Stem the list of words
        filtered_words_list = [stem(word) for word in filtered_words_list]
        # Word ranking
        ranked_dict = dict()
        for word in filtered_words_list:
            if word in ranked_dict:
                ranked_dict[word] += 1
            else:
                ranked_dict[word] = 1
        return ranked_dict


    @staticmethod
    def process_text_query_plain(input_text: str):
        """
        Runs text processing on a given text
        :param input_text: String
        :return: processed string
        """
        # Convert characters to lower case
        input_text_to_lower = input_text.lower()
        # Remove special characters from the string
        input_text_to_lower = re.sub('[^a-zA-Z0-9 \n]', '', input_text_to_lower)
        # Remove common words using list of stop words
        filtered_words_list = [word for word in input_text_to_lower.split() if word not in Ranking.stop_words]
        # Stem the list of words
        filtered_words_list = [stem(word) for word in filtered_words_list]
        # Merge back togetheer
        return " ".join(filtered_words_list)

    def gather_queries(self):
        """
        Gathers the queries
        :return: query_structure (query_id_plain, query_id_formatted, Ranking.process_text_query(query_id_plain) list 
        """
        query_tup_list = []
        for page in self.pages:
            for section_path in page.flat_headings_list():
                query_id_plain = " ".join([page.page_name] + [section.heading for section in section_path])
                query_id_formatted = "/".join([page.page_id] + [section.headingId for section in section_path])
                tup = (query_id_plain, query_id_formatted, Ranking.process_text_query(query_id_plain))
                query_tup_list.append(tup)
        if self.enable_cache is True:
            print(" Query cache created successfully ")
            _pickle.dump(query_tup_list, open(os.path.join(os.curdir, "cache/query_structure_cache"), "wb"))
        return query_tup_list

    def gather_queries_and_page(self):
        """
        Gathers the queries and corresponding pagename
        :return: query_structure (query_id_plain, query_id_formatted, Ranking.process_text_query(query_id_plain, page.page_name) list 
        """
        query_tup_list = []
        for page in self.pages:
            for section_path in page.flat_headings_list():
                query_id_plain = " ".join([page.page_name] + [section.heading for section in section_path])
                query_id_formatted = "/".join([page.page_id] + [section.headingId for section in section_path])
                tup = (query_id_plain, query_id_formatted, Ranking.process_text_query(query_id_plain), page.page_name)
                query_tup_list.append(tup)
        return query_tup_list

    @staticmethod
    def process_text_append_text_annotations(input_text: str):
        # Find annotations in a text
        annotations = tagme.annotate(input_text, GCUBE_TOKEN)
        entities = " ".join([word.entity_title for word in annotations.get_annotations(0.2)])
        # Convert characters to lower case
        input_text_to_lower = (input_text + " " + entities).lower()
        # Remove special characters from the string
        input_text_to_lower = re.sub('[^a-zA-Z0-9 \n]', '', input_text_to_lower)
        # Remove common words using list of stop words
        filtered_words_list = [word for word in input_text_to_lower.split() if word not in Ranking.stop_words]
        # Stem the list of words
        filtered_words_list = [stem(word) for word in filtered_words_list]
        # Word ranking
        ranked_dict = dict()
        for word in filtered_words_list:
            if word in ranked_dict:
                ranked_dict[word] += 1
            else:
                ranked_dict[word] = 1
        return ranked_dict

    @staticmethod
    def process_text_append_text_mentions(input_text: str):
        # Find spots in a text
        mentions = tagme.mentions(input_text, GCUBE_TOKEN)
        entities = " ".join([word.mention for word in mentions.get_mentions(0.01)])
        # Convert characters to lower case
        input_text_to_lower = (input_text + " " + entities).lower()
        # Remove special characters from the string
        input_text_to_lower = re.sub('[^a-zA-Z0-9 \n]', '', input_text_to_lower)
        # Remove common words using list of stop words
        filtered_words_list = [word for word in input_text_to_lower.split() if word not in Ranking.stop_words]
        # Stem the list of words
        filtered_words_list = [stem(word) for word in filtered_words_list]
        # Word ranking
        ranked_dict = dict()
        for word in filtered_words_list:
            if word in ranked_dict:
                ranked_dict[word] += 1
            else:
                ranked_dict[word] = 1
        return ranked_dict

    def gather_entity_enhanced_queries_mentions(self):
        query_tup_list = []
        for page in self.pages:
            for section_path in page.flat_headings_list():
                query_id_plain = " ".join([page.page_name] + [section.heading for section in section_path])
                query_id_formatted = "/".join([page.page_id] + [section.headingId for section in section_path])
                enhance_query_list = Ranking.process_text_append_text_mentions(query_id_plain)
                tup = (query_id_plain, query_id_formatted, enhance_query_list)
                print(tup)
                query_tup_list.append(tup)
        if self.enable_cache is True:
            print(" Query cache created successfully ")
            _pickle.dump(query_tup_list, open(os.path.join(os.curdir, "cache/query_structure_cache"), "wb"))
        return query_tup_list

    def gather_entity_enhanced_paragraphs_mentions(self):
        id_to_text_dict = dict()
        with open(self.paragraph_file, 'rb') as f:
            for p in itertools.islice(iter_paragraphs(f), 0, self.passages_extract):
                id_to_text_dict[p.para_id] = Ranking.process_text_append_text_mentions(p.get_text())
                print(id_to_text_dict[p.para_id])
        if self.enable_cache is True:
            print(" Paragraph cache created successfully ")
            _pickle.dump(id_to_text_dict, open(os.path.join(os.curdir, "cache/paragraph_structure"), "wb"))
        return id_to_text_dict

    def gather_entity_enhanced_queries_annotations(self):
        query_tup_list = []
        for page in self.pages:
            for section_path in page.flat_headings_list():
                query_id_plain = " ".join([page.page_name] + [section.heading for section in section_path])
                query_id_formatted = "/".join([page.page_id] + [section.headingId for section in section_path])
                enhance_query_list = Ranking.process_text_append_text_annotations(query_id_plain)
                tup = (query_id_plain, query_id_formatted, enhance_query_list )
                query_tup_list.append(tup)
        if self.enable_cache is True:
            print(" Query cache created successfully ")
            _pickle.dump(query_tup_list, open(os.path.join(os.curdir, "cache/query_structure_cache"), "wb"))
        return query_tup_list

    def gather_entity_enhanced_paragraphs_annotations(self):
        id_to_text_dict = dict()
        with open(self.paragraph_file, 'rb') as f:
            for p in itertools.islice(iter_paragraphs(f), 0, self.passages_extract):
                id_to_text_dict[p.para_id] = Ranking.process_text_append_text_annotations(p.get_text())
        if self.enable_cache is True:
            print(" Paragraph cache created successfully ")
            _pickle.dump(id_to_text_dict, open(os.path.join(os.curdir, "cache/paragraph_structure"), "wb"))
        return id_to_text_dict
