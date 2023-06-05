from __future__ import annotations
import sys
import logging.handlers
import src.propaganda_techniques as pt
import src.annotation as an

__author__ = "Giovanni Da San Martino"
__copyright__ = "Copyright 2019"
__credits__ = ["Giovanni Da San Martino"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Giovanni Da San Martino"
__email__ = "gmartino@hbku.edu.qa"
__status__ = "Beta"

logger = logging.getLogger("propaganda_scorer")


class AnnotationWithConfidence(an.Annotation):

    """
    One annotation is represented by 
        - a span (two integer indices indicating the starting and ending position of the span), 
        - the propaganda technique name (a label attached to the span) 
        - a confidence for the prediction. The confidence is treated as an optional argument (when not provided, the default value DEFAULT_CONFIDENCE is used). 

    The class provides basic maniputation functions for one annotation. 
    """

    # input file format variables
    separator = "\t"
    ARTICLE_ID_COL = 0
    TECHNIQUE_NAME_COL = 1
    FRAGMENT_START_COL = 2
    FRAGMENT_END_COL = 3
    CONFIDENCE_COL = 4
    DEFAULT_CONFIDENCE = 1.0
    propaganda_techniques:pt.Propaganda_Techniques = None


    def __init__(self, label:str=None, start_offset:str = None, end_offset:str=None, confidence:float = None): 
        
        super().__init__(label, start_offset, end_offset)
        self.confidence = confidence


    def __str__(self):

        return super().__str__() + " (confidence: " + str(self.confidence) +  ")"


    def get_confidence(self)->float:

        return self.confidence
    

    def set_confidence(self, new_confidence:float)->None:

        self.confidence = new_confidence


    @staticmethod
    def load_annotation_from_string(annotation_string:str, row_num:int=None, filename:str=None)->(Annotation, str):
        """
        Read annotations from a csv-like string, with fields separated
        by the class variable `separator`: 

        article id<separator>technique name<separator>starting_position<separator>ending_position
        Fields order is determined by the class variables ARTICLE_ID_COL,
        TECHNIQUE_NAME_COL, FRAGMENT_START_COL, FRAGMENT_END_COL, CONFIDENCE_COL

        Besides reading the data, it performs basic checks.

        :return a tuple (AnnotationWithConfidence object, id of the article)
        """

        row = annotation_string.rstrip().split(AnnotationWithConfidence.separator)
        if len(row) < 4:
            logger.error("Row%s%s is supposed to have at least 4 columns. Found %d: -%s-."
                         % (" " + str(row_num) if row_num is not None else "",
                            " in file " + filename if filename is not None else "", len(row), annotation_string))
            sys.exit()

        article_id = row[AnnotationWithConfidence.ARTICLE_ID_COL]
        label = row[AnnotationWithConfidence.TECHNIQUE_NAME_COL]
        try:
            start_offset = int(row[AnnotationWithConfidence.FRAGMENT_START_COL])
        except:
            logger.error("The column %d in row%s%s is supposed to be an integer (found %s): -%s-"
                         %(AnnotationWithConfidence.FRAGMENT_START_COL, " " + str(row_num) if row_num is not None else "", " in file " + filename if filename is not None else "", row[AnnotationWithConfidence.FRAGMENT_START_COL], annotation_string))
        try:
            end_offset = int(row[AnnotationWithConfidence.FRAGMENT_END_COL])
        except:
            logger.error("The column %d in row%s%s is supposed to be an integer (found %s): -%s-"
                         %(AnnotationWithConfidence.FRAGMENT_END_COL, " " + str(row_num) if row_num is not None else "", " in file " + filename if filename is not None else "", row[AnnotationWithConfidence.FRAGMENT_END_COL], annotation_string))
        try:
            confidence = float(row[AnnotationWithConfidence.CONFIDENCE_COL])
        except:
            confidence = AnnotationWithConfidence.DEFAULT_CONFIDENCE
            #logger.warning("Confidence not provided (setting to default value %f) in column %d, row%s%s: -%s-"
            #            %(AnnotationWithConfidence.DEFAULT_CONFIDENCE, AnnotationWithConfidence.CONFIDENCE_COL, " " + str(row_num) if row_num is not None else "", " in file " + filename if filename is not None else "", annotation_string))
            
        return AnnotationWithConfidence(label, start_offset, end_offset, confidence), article_id
        

    def check_format_of_annotation_in_file(self):
        """
        Performs some checks on the fields of the annotation. 
        Note that the confidence is treated as an optional argument. 
        """
        if not self.is_technique_name_valid():
            return False
        if not self.is_span_valid():
            return False
        return True



    def merge_spans(self, second_annotation:AnnotationWithOutLabel)->None:
        """
        Merge the spans of two annotations. The function does not check whether the spans overlap. 
        The confidence becomes the average of the two confidence values

        :param second_annotation: the AnnotationWithConfidence object whose span is being merged
        :return: None, the current annotation is modified in place (no change to second_annotation)
        """
        super().merge_spans(second_annotation)
        self.set_confidence((self.confidence + second_annotation.confidence)/2)
        
