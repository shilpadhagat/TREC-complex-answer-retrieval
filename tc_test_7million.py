"""
Note: This is a static code written only for the purposes of testing 7 million on server
For actual file generation refer to regular parameter taking files like tc_rerank_document_framework.py ( Git readme )
"""


from tc_modified_ranking_7million import PartialRanking

ranking_object = PartialRanking("test.benchmarkY1.omit.cbor.outlines", "all.test200.cbor.paragraphs", 7000000)

ranking_object.gather_queries()
#ranking_object.gather_paragraphs()

