import re
import tagme

from trec_car.read_data import *
from nltk.corpus import stopwords
from stemming.porter2 import stem


"""
@author: Shilpa Dhagat.
Contain methods to generate the query and paragraph structure used by the relevance model.
"""

GCUBE_TOKEN = "bfbfb535-3683-47c0-bd11-df06d5d96726-843339462"
DEFAULT_LANG = "en"
DEFAULT_TAG_API = "https://tagme.d4science.org/tagme/tag"
DEFAULT_SPOT_API = "https://tagme.d4science.org/tagme/spot"
DEFAULT_REL_API = "https://tagme.d4science.org/tagme/rel"


class EntityLinkingAndRelevance:

    stop_words = stopwords.words('english')

    def __init__(self, outline_file, paragraph_file, output_file):
        self.outline_file = outline_file
        self.paragraph_file = paragraph_file
        self.output_file = output_file
        self.pages = self.gather_pages()
        self.queries = self.gather_queries()
        self.paragraphs = self.gather_paragraphs()
        self.enhanced_queries = self.gather_entity_enhanced_queries_mentions()
        self.enhanced_paragraphs = self.gather_entity_enhanced_paragraphs_mentions()

    def gather_pages(self):
        with open(self.outline_file, 'rb') as f:
            pages = [p for p in itertools.islice(iter_annotations(f), 0, 1000)]
        return pages

    def gather_paragraphs(self):
        id_to_text_dict = dict()
        with open(self.paragraph_file, 'rb') as f:
            for p in itertools.islice(iter_paragraphs(f), 0, 4000):
                id_to_text_dict[p.para_id] = EntityLinkingAndRelevance.process_text(p.get_text())
        return id_to_text_dict

    @staticmethod
    def process_text(input_text: str):
        # Convert characters to lower case
        input_text_to_lower = input_text.lower()
        # Remove special characters from the string
        input_text_to_lower = re.sub('[^a-zA-Z0-9 \n]', '', input_text_to_lower)
        # Remove common words using list of stop words
        filtered_words_list = [word for word in input_text_to_lower.split() if word not in EntityLinkingAndRelevance.stop_words]
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

    def gather_queries(self):
        query_tup_list = []
        for page in self.pages:
            for section_path in page.flat_headings_list():
                query_id_plain = " ".join([page.page_name] + [section.heading for section in section_path])
                query_id_formatted = "/".join([page.page_id] + [section.headingId for section in section_path])
                tup = (query_id_plain, query_id_formatted, EntityLinkingAndRelevance.process_text(query_id_plain))
                query_tup_list.append(tup)
                #print(tup)
        return query_tup_list

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
        filtered_words_list = [word for word in input_text_to_lower.split() if word not in EntityLinkingAndRelevance.stop_words]
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
        enhance_query_list = []
        for page in self.pages:
            for section_path in page.flat_headings_list():
                query_id_plain = " ".join([page.page_name] + [section.heading for section in section_path])
                enhance_query_list = EntityLinkingAndRelevance.process_text_append_text_mentions(query_id_plain)
        return enhance_query_list

    def gather_entity_enhanced_paragraphs_mentions(self):
        id_to_text_dict = dict()
        with open(self.paragraph_file, 'rb') as f:
            for p in itertools.islice(iter_paragraphs(f), 0, 4000):
                id_to_text_dict[p.para_id] = EntityLinkingAndRelevance.process_text_append_text_mentions(p.get_text())
                print(id_to_text_dict)
        return id_to_text_dict


    def get_queries(self):
        return self.queries

    def get_paragraphs(self):
        return self.paragraphs

    def get_enhanced_queries(self):
        return self.enhanced_queries

    def get_enhanced_paragraphs(self):
        return self.enhanced_paragraphs

