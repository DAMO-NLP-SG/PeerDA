import glob
import os.path
import re
import sys
import argparse
import logging.handlers
import src.annotations as anns
import src.annotation as an
import src.propaganda_techniques as pt

def main(args):

	articles_folder = args.articles_folder
	annotations_folder = args.annotations_folder
	propaganda_techniques_file = args.propaganda_techniques_file
	check_duplicated_annotations = args.check_duplicated_annotation
	check_out_of_boundaries_annotations = args.check_out_of_boundaries_annotation
	check_start_end_chars_of_annotations = args.check_start_end_chars_of_annotation
	output_file_suffix = args.output_file_suffix
	fix_errors = args.fix_errors

	logger = logging.getLogger("propaganda_scorer")
	ch = logging.StreamHandler(sys.stdout)
	ch.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	ch.setFormatter(formatter)
	logger.setLevel(logging.DEBUG)
	logger.addHandler(ch)

	if not os.path.exists(articles_folder) or not os.path.exists(articles_folder):
		logger.error("trying to load articles from folder %s, which does not exists"%(articles_folder))
		sys.exit()
	if not os.path.isdir(articles_folder):
		logger.error("trying to load articles from folder %s, which does not appear to be a valid folder"%(articles_folder))
		sys.exit()

	annotations = anns.Annotations()
	an.Annotation.set_propaganda_technique_list_obj(pt.Propaganda_Techniques(filename=propaganda_techniques_file))
	if not annotations.load_annotation_list_from_folder(annotations_folder):
		sys.exit("Quitting...")
	
	regex = re.compile("article([0-9]+).*")

	for article in glob.glob(os.path.join(articles_folder, "*.txt")):

		article_id = regex.match(os.path.basename(article)).group(1)
		if article_id not in annotations.get_article_id_list():
			sys.exit("Article id %s not found"%(article_id))
		with open(article, "r") as f:
			article_content = f.read()
		aa_obj = annotations.get_article_annotations_obj(article_id)
		article_correct, needsaving = aa_obj.check_article_annotations(article_content, fix_errors, 
									       check_out_of_boundaries_annotations,
									       check_start_end_chars_of_annotations,
									       check_duplicated_annotations)
		if not article_correct:
			logger.info("article %s: %s"%(article_id, "OK" if article_correct else "ERROR"))
		if needsaving:
			output_file = annotations_folder + "/article" + article_id + output_file_suffix
			aa_obj.set_output_format()
			aa_obj.save_annotations_to_file(output_file)
			logger.info("article saved to file %s"%(output_file))


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Performs some checks on a corpus. \n" +
									 "Example: check_corpus_annotations.py -s data/article736757214.task-FLC.labels -t data/article736757214.txt")
	parser.add_argument('-t', '--articles-folder', dest='articles_folder', required=True, help="path to the folder with articles", default="data/articles/")
	parser.add_argument('-s', '--annotations-folder', dest='annotations_folder', required=True,
						help="folders with annotation files.", default="data/articles_FLC_annotations")
	parser.add_argument('-o', '--output-file-suffix', dest='output_file_suffix', required=False, 
			help="suffix of output files. Output file name will be [annotations_folder]/article[article_id][suffix]",default=".task2-TC.labels.fix")
	parser.add_argument('-p', '--propaganda-techniques-list-file', dest='propaganda_techniques_file', required=False, 
			help="path to the file with the list of propaganda techniques", default="data/propaganda-techniques-names-semeval2020task11.txt")
	parser.add_argument('-d', '--check-duplicated-annotations', dest='check_duplicated_annotation', required=False, action='store_true', help="Check for duplicated annotations in the same article.")
	parser.add_argument('-b', '--check-out-of-boundaries-annotations', dest='check_out_of_boundaries_annotation', required=False,
						action='store_true', help="Check for annotlsations that refer to spans out of the boundaries of the article.")
	parser.add_argument('-c', '--check-start-end-chars-of-annotations', dest='check_start_end_chars_of_annotation', required=False, action='store_true', help="Check that annotations do not start/end with a specific set of chars (see code).")
	parser.add_argument('-f', '--fix-errors', dest='fix_errors', required=False, action='store_true', help="Fix errors whenever possible.")

	main(parser.parse_args())
