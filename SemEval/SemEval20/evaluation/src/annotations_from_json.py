from __future__ import annotations
from typing import Dict
import sys
import json
import os.path
import logging.handlers
import src.annotation as an
import src.annotations as ans

__author__ = "Giovanni Da San Martino"
__copyright__ = "Copyright 2019"
__credits__ = ["Giovanni Da San Martino"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Giovanni Da San Martino"
__email__ = "dasan@math.unipd.it"
__status__ = "Beta"

logger = logging.getLogger("propaganda_scorer")

class AnnotationsFromJson(ans.Annotations):
    """
    Dictionary of Articles_annotations objects loaded from a json file 
    (basically a dataset of article_annotations objects)

    """

    def check_annotation_spans_with_category_matching(self, merge_overlapping_spans:bool=False):
        """
        Check whether there are overlapping spans for the same technique in the same article.
        Two spans are overlapping if their associated techniques match (according to category_matching_func)
        If merge_overlapping_spans==True then the overlapping spans are merged, otherwise an error is raised.

        :param merge_overlapping_spans: if True merges the overlapping spans
        :return:
        """

        for article_id in self.get_article_id_list():

            annotation_list = self.get_article_annotations_obj(article_id).groupby_technique()
            if merge_overlapping_spans:
                for technique in annotation_list.keys():
                    for i in range(1, len(annotation_list[technique])):
                        annotation_list[technique][i].merge_spans(annotation_list[technique], i-1)
            if not self.get_article_annotations_obj(article_id):
                return False
        return True


    def load_annotation_list_from_file(self, filename):
        """
        Loads all annotations in json file <filename>. The file is supposed to contain annotations for multiple articles. 
        The json file is a list of dictionaries (one per example). 
        Each annotation is checked according to check_format_of_annotation_in_file()
        """
        try:
            with open(filename) as p:
                submission = json.load(p)
        except:
            sys.exit(logger.error("File is not a valid json file: {}".format(filename)))
        
        error=False
        KEYS = ['id', 'labels']
        for i, obj in enumerate(submission):
            for key in KEYS:
                if key not in obj:
                    logger.error("Missing entry in line {}:{}".format(i, key))
                    error = True
                    break
            self.create_article_annotations_object(obj['id']) #creates an article even when it has no annotations
            for j, label in enumerate(obj['labels']):
                for fieldname in ('technique', 'start', 'end'):
                    if fieldname not in label.keys():
                        logger.error("Missing field %s in example %d label %d\n%s"%(fieldname,i,j,str(label)))
                        error = True
                        break
                if error==True:
                    sys.exit(logger.error("Annotation format not valid in file %s"%(filename)))
                else:
                    ann = an.Annotation(label['technique'],label['start'], label['end'])
                    if not ann.check_format_of_annotation_in_file():
                        sys.exit(logger.error("Annotation %d of example %s format not valid in file %s"%(j, obj['id'], filename)))
                    self.add_annotation(ann, obj['id'])
            


