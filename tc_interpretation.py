"""
@author: Shilpa Dhagat.

"""

import argparse
from tc_interpret_entitylinking import InterpretEntityLinking


class GIF(object):
    """
 
        query_annots: candidate entity ranking annotations for a query.
    """

    def __init__(self, query_annot):
        self.score_th = 2
        self.query_annot = query_annot

    def process_query(self):
        """
        Processesing w.r.t scores is not required since it is already been filtered in the tagme.annotate method.
        Score for all the annotations > 0.2
        Takes query annotations and generates the interpretation sets.

        """

        interprets = self.form_interprets()
        return interprets

    def form_interprets(self):
            """
            Forms query interpretations from the given annotations.
            :returns list of query interpretations [{men1: en1, ..}, ..]
            """
            query_interpret = [{}]
            for mention, entity in self.query_annot:
                    added = False
                    for interpret in query_interpret:
                            mentions = interpret.keys()
                            mentions.append(mention)
                            if not self.__is_overlapping(mentions):
                                    interpret[mention] = entity
                                    added = True
                    if not added:
                            query_interpret.append({mention: entity})
            return query_interpret


def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("outline_file", type=str, help="Qualified location of the outline file")
        parser.add_argument("paragraph_file", type=str, help="Qualified location of the paragraph file")
        parser.add_argument("output_file", type=str, help="Name of the output file")
        parser.add_argument("-th", "--threshold", help="Score threshold for greedy approach", default=None, type=float)
        args = vars(parser.parse_args())

        query_cbor = args['outline_file']
        paragraphs_cbor = args['paragraph_file']
        output_file_name = args['output_file']
        threshhold = 20

        ranking = InterpretEntityLinking(query_cbor, paragraphs_cbor, output_file_name)

        query_annotations = ranking.gather_entity_enhanced_queries_annotations()
        #print(query_annotations)
        interprets = {}
        for query_annot in query_annotations:
                print(query_annot)
                interprets[query_annot] = GIF(query_annot).process_query()


if __name__ == "__main__":
    main()
