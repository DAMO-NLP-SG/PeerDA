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
import src.article_annotations as an


def main(args):

    span_file = args.spans_file
    write_technique_on_output = bool(args.write_technique_on_output)
    prop_vs_non_propaganda = bool(args.fragments_only)
    output_file = args.output_file
    if not output_file:
        output_file = span_file + ".merged"


    article_annotations = an.Articles_annotations()
    article_annotations.load_article_annotations_from_csv_file(span_file)
    article_annotations.has_overlapping_spans(prop_vs_non_propaganda, True)
    article_annotations.set_output_format(True, True, write_technique_on_output)
    article_annotations.save_annotations_to_file(output_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Merge annotations in a single annotation file.") 
    parser.add_argument('-s', '--spans-file', dest='spans_file', required=True, 
                        help="file with annotations. ")
    parser.add_argument('-o', '--output-file', dest='output_file', required=False, default="", 
                        help="name of the output file. If not specified: [--spans-file].merged")
    parser.add_argument('-l', '--write-techniques-to-output-file', dest='write_technique_on_output', required=False,
                        action='store_true', help="Write technique name on output.")
    parser.add_argument('-f', '--fragments-only', dest='fragments_only', required=False, action='store_true', default=False,
                        help="If the option is added, two fragments match independently of their propaganda techniques")
    main(parser.parse_args())

