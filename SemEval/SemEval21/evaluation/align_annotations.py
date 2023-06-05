__author__ = "Giovanni Da San Martino"
__copyright__ = "Copyright 2019"
__credits__ = ["Giovanni Da San Martino"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Giovanni Da San Martino"
__email__ = "gmartino@hbku.edu.qa"
__status__ = "Beta"

import codecs
import argparse
import copy
import src.annotation_w_o_label as an
import src.article_annotations as aa


def main(args):

    span_file = args.spans_file
    article_file = args.article_file
    original_text_file = args.original_text_file
    gold_spans = args.gold_spans
    debug = args.debug
    if args.output_file_name is None:
        output_file_name = span_file + ".aligned.txt"
    else:
        output_file_name = args.output_file_name

    annotations = aa.Articles_annotations()
    annotations.load_article_annotations_from_csv_file(span_file, an.AnnotationWithOutLabel)
    annotations.sort_spans()
    annotations.has_overlapping_spans(True, True)

    with codecs.open(article_file, "r", encoding="utf8") as f:
        article_content = f.read()

    with codecs.open(original_text_file, "r", encoding="utf8") as f:
        original_text = f.read()

    if debug==True:
        gold_annotations = aa.Articles_annotations()
        gold_annotations.load_article_annotations_from_csv_file(gold_spans, an.AnnotationWithOutLabel)
        gold_annotations.sort_spans()
        gold_annotations_content = gold_annotations.get_spans_content(original_text)
        print(gold_annotations)
        print("gold annotations content (no spaces):\n%s\n---"%(gold_annotations_content))
        before = annotations.get_spans_content(article_content)
        print(annotations)
        print("content of annotations of converted text:\n%s\n---"%(before))

    original_annotations = copy.deepcopy(annotations)
    annotations.align_annotation_to_new_text(original_text, article_content)
    after = annotations.get_spans_content(original_text)
    
    annotations.set_output_format(True, True, False)
    annotations.save_annotations_to_file(output_file_name)

    if gold_spans is not None:
        gold_annotations = aa.Articles_annotations()
        gold_annotations.load_article_annotations_from_csv_file(gold_spans, an.AnnotationWithOutLabel)
        gold_annotations.sort_spans()
        gold_annotations.has_overlapping_spans(True, True)
        annotations.sort_spans()
        if not gold_annotations==annotations:
            print("%s: The following annotations are different: "%(article_file),end=""), 
            for p in annotations-gold_annotations:
                print("1: %s; 2: %s -- %s -- %s"%(p[0], p[1], p[0].get_span_content(article_content), p[1].get_span_content(original_text))),
            print("%s: The following annotations are problematic:\n(USER ANNOTATION) %s\n(GOLD ANNOTATION) %s\n"%(article_file, annotations, gold_annotations))
            gold_annotations_content = gold_annotations.get_spans_content(original_text)
            print("gold annotations content (no spaces):\n%s\n---"%(gold_annotations_content))
            before = original_annotations.get_spans_content(article_content)
            print("content of annotations of converted text before alignment:\n%s\n---"%(before))
        else:
            print("OK: %s -> %s"%(article_file, output_file_name))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create a copy of the annotations such that they refer to    new_text (a slightly modified version of original_text, the text the current annotation refers to). The new text currently it is supposed to differ from the original text only by spaces\n" + 
                                     "Example: python align_annotations.py -t data/article727497152.txt.converted -s data/article727497152.task1-SI.labels.converted -o data/article727497152.txt -g data/article727497152.task1-SI.labels")
    parser.add_argument('-t', '--text-file-converted', dest='article_file', required=True, help="text document file without added spaces and transformed chars")
    parser.add_argument('-s', '--spans-file-converted', dest='spans_file', required=True, 
                        help="file with the list of annotations referring to the original text file.")
    parser.add_argument('-o', '--original-text-file', dest='original_text_file', required=True, 
                        help="text file whose --gold-spans annotations refers to. The file has added spaces and transformed chars")
    parser.add_argument('-g', '--gold-spans-original', dest='gold_spans', required=False, default=None, 
                        help="file with target spans related to --text-file. This parameter is for debugging only")
    parser.add_argument('-n', '--output-annotation-file-name', dest='output_file_name', required=False, 
                        default=None, help="file name which aligned annotations will be saved to")
    parser.add_argument('-d', '--enable-debug', dest='debug', required=False, default=False,                                      action='store_true', help="enable printing of debugging information")

    main(parser.parse_args())
