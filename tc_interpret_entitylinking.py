"""
@author: Shilpa Dhagat.

"""

from trec_car.read_data import *
from nltk.corpus import stopwords
import re
from stemming.porter2 import stem
import tagme

GCUBE_TOKEN = "bfbfb535-3683-47c0-bd11-df06d5d96726-843339462"
DEFAULT_LANG = "en"
DEFAULT_TAG_API = "https://tagme.d4science.org/tagme/tag"
DEFAULT_SPOT_API = "https://tagme.d4science.org/tagme/spot"
DEFAULT_REL_API = "https://tagme.d4science.org/tagme/rel"


class InterpretEntityLinking:

    stop_words = stopwords.words('english')

    def __init__(self, outline_file, paragraph_file, output_file):
        self.outline_file = outline_file
        self.paragraph_file = paragraph_file
        self.output_file = output_file
        self.pages = self.gather_pages()
        # self.queries = self.gather_entity_enhanced_queries_annotations()

    def gather_pages(self):
        with open(self.outline_file, 'rb') as f:
            pages = [p for p in itertools.islice(iter_annotations(f), 0, 1000)]
        return pages

    @staticmethod
    def process_text_append_text_annotations(input_text: str):
        entity_mention_tup_list = []
        annotations = tagme.annotate(input_text, GCUBE_TOKEN)
        entities = " ".join([word.entity_title for word in annotations.get_annotations(0.2)])
        mentions = " ".join([word.mention for word in annotations.get_annotations(0.2)])
        # scores = " ".join([word.score for word in annotations.get_annotations(0.2)])

        # Convert characters to lower case
        entity_to_lower = (input_text + " " + entities).lower()
        # Remove special characters from the string
        entity_to_lower = re.sub('[^a-zA-Z0-9 \n]', '', entity_to_lower)
        # Remove common entities using list of stop words
        filtered_entities_list = [word for word in entity_to_lower.split() if word not in InterpretEntityLinking.stop_words]
        # Stem the list of entites
        filtered_entities_list = [stem(word) for word in filtered_entities_list]

        # Convert characters to lower case
        mention_to_lower = (input_text + " " + mentions).lower()
        # Remove special characters from the string
        mention_to_lower = re.sub('[^a-zA-Z0-9 \n]', '', mention_to_lower)
        # Remove common mentions using list of stop words
        filtered_mentions_list = [word for word in mention_to_lower.split() if
                               word not in InterpretEntityLinking.stop_words]
        # Stem the list of mentions
        filtered_mentions_list = [stem(word) for word in filtered_mentions_list]

        tup = (filtered_entities_list, filtered_mentions_list)
        entity_mention_tup_list.append(tup)
        return entity_mention_tup_list

    def gather_entity_enhanced_queries_annotations(self):
        query_tup_list = []
        for page in self.pages:
            for section_path in page.flat_headings_list():
                query_id_plain = " ".join([page.page_name] + [section.heading for section in section_path])
                enhance_query_list = InterpretEntityLinking.process_text_append_text_annotations(query_id_plain)
                # print(tup)
                query_tup_list.append(enhance_query_list)
                #print(query_tup_list)
        return query_tup_list

    def gather_entity_enhanced_paragraphs_annotations(self):
        id_to_text_dict = dict()
        with open(self.paragraph_file, 'rb') as f:
            for p in itertools.islice(iter_paragraphs(f), 0, 4000):
                id_to_text_dict[p.para_id] = InterpretEntityLinking.process_text_append_text_annotations(p.get_text())
                # print(id_to_text_dict[p.para_id])
        return id_to_text_dict

    def get_queries(self):
        return self.queries

    def get_paragraphs(self):
        return self.paragraphs
