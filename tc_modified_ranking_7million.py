import _pickle
import os
import re
from nltk.corpus import stopwords
from stemming.porter2 import stem
from trec_car.read_data import *
import gc

class PartialRanking:
    cache_words = dict()

    def __init__(self, outline_file, paragraph_file, no_passages_to_extract):
        """
        Constructor
        :param outline_file: path of the outline file
        :param paragraph_file: path of the paragraph file
        """
        self.passages_extract = no_passages_to_extract
        self.outline_file = outline_file
        self.paragraph_file = paragraph_file
        self.pages = self.gather_pages()
        self.stop_words = stopwords.words('english')

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
        Modified for running 
        """
        id_to_text_dict = dict()
        with open(self.paragraph_file, 'rb') as f:
            counter = 0
            inner_counter = 0
            for p in itertools.islice(iter_paragraphs(f), 0, self.passages_extract):
                id_to_text_dict[p.para_id] = self.process_text_query(p.get_text())
                counter += 1
                if counter == 70000:
                    name_of_file = "para_collection" + str(inner_counter)
                    _pickle.dump(id_to_text_dict, open(os.path.join(os.curdir, "merge_cache/"+name_of_file), "wb"))
                    print("Created pickle dump for collection named " + name_of_file)
                    id_to_text_dict.clear()
                    inner_counter += 1
                    counter = 0
                    gc.collect()

    def process_text_query(self, input_text: str):
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
        filtered_words_list = [word for word in input_text_to_lower.split() if word not in self.stop_words]
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
        """
        Modified for running
        """
        query_tup_list = []
        for page in self.pages:
            for section_path in page.flat_headings_list():
                query_id_plain = " ".join([page.page_name] + [section.heading for section in section_path])
                query_id_formatted = "/".join([page.page_id] + [section.headingId for section in section_path])
                tup = (query_id_plain, query_id_formatted, self.process_text_query(query_id_plain))
                query_tup_list.append(tup)
        _pickle.dump(query_tup_list, open(os.path.join(os.curdir, "cache/test_queries"), "wb"))
