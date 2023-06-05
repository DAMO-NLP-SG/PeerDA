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
import src.article_annotations as aa
import src.annotation as an
import src.propaganda_techniques as pt

def main(args):

    span_file = args.spans_file
    article_file = args.article_file
    print_line_numbers = bool(args.add_line_numbers)
    fix_from_char_index = int(args.fix_from_char_index)
    offset = int(args.offset)
    propaganda_techniques_file = args.propaganda_techniques_file

    annotations = aa.Articles_annotations()
    an.Annotation.set_propaganda_technique_list_obj(pt.Propaganda_Techniques(filename=propaganda_techniques_file))

    annotations.load_article_annotations_from_csv_file(span_file)
    annotations.shift_spans(fix_from_char_index, offset)

    with codecs.open(article_file, "r", encoding="utf8") as f:
        article_content = f.read()
    
    output_text, footnotes, legend = annotations.mark_text(article_content, print_line_numbers)

    print(output_text)
    print(legend)
    print(footnotes)

    if offset != 0:
        annotations.save_annotations_to_file(span_file + ".fix")
        print("Fixed annotations saved to file %s.fix"%(span_file))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Highlight labelled spans in a text file. Do not pipe with less. \n" + 
                                     "Example: print_spans.py -s data/article736757214.task-FLC.labels -t data/article736757214.txt")
    parser.add_argument('-t', '--text-file', dest='article_file', required=True, help="file with text document")
    parser.add_argument('-s', '--spans-file', dest='spans_file', required=True, 
                        help="file with spans to be highlighted. One line of the span file")
    parser.add_argument('-l', '--add-line-numbers', dest='add_line_numbers', required=False,
                        action='store_true', help="Prepend line numbers on output.")
    parser.add_argument('-f', '--fix-from-char-index', dest='fix_from_char_index', required=False, default=0, 
                        help="Starting char index from which annotations are shifted by offset.")
    parser.add_argument('-o', '--offset', dest='offset', required=False, default=0, 
                        help="Offset by which annotations are shifted.")
    parser.add_argument('-p', '--propaganda-techniques-list-file', dest='propaganda_techniques_file',required=False, help="path to the file with the list of propaganda techniques", default="data/propaganda-techniques-names-semeval2020task11.txt")

    main(parser.parse_args())
