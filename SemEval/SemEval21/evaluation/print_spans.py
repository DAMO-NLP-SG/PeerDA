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
import src.annotation as an
import src.article_annotations as aa
import src.propaganda_techniques as pt
import logging.handlers


def main(args):

    span_file = args.spans_file
    article_file = args.article_file
    propaganda_techniques_list_file = args.propaganda_techniques_list_file
    debug_on_std = bool(args.debug_on_std)

    if not debug_on_std:
        logging.getLogger("propaganda_scorer").setLevel(logging.ERROR)

    propaganda_techniques = pt.Propaganda_Techniques(propaganda_techniques_list_file)
    annotations = aa.Articles_annotations()
    aa.Articles_annotations.techniques = propaganda_techniques

    annotations.load_article_annotations_from_csv_file(span_file)
    
    with codecs.open(article_file, "r", encoding="utf8") as f:
        article_content = f.read()
    
    #print("\n".join([str(i)+") "+x for i,x in enumerate(str(aa.techniques).split("\n"))]))
    output_text, footnotes = annotations.tag_text_with_annotations(article_content) #add html tags
    #output_text, footnotes, legend = annotations.mark_text(article_content)    #mark annotations for terminal

    print(output_text)
    print(footnotes)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Add tags to mark spans in a text file. \n" + 
                                     "Example: print_spans.py -s data/article736757214.task-FLC.labels -t data/article736757214.txt")
    parser.add_argument('-t', '--text-file', dest='article_file', required=True, help="file with text document")
    parser.add_argument('-s', '--spans-file', dest='spans_file', required=True, 
                        help="file with spans to be highlighted. One line of the span file")
    parser.add_argument('-p', '--propaganda-techniques-list-file', dest='propaganda_techniques_list_file', required=False, 
                        default="data/propaganda-techniques-names.txt", 
                        help="file with list of propaganda techniques (one per line).")
    parser.add_argument('-d', '--enable-debug-on-standard-output', dest='debug_on_std', required=False,
                        action='store_true', help="Print debug info on standard output.")
 
    main(parser.parse_args())
