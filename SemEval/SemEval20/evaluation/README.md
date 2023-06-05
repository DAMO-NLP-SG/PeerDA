
Scorers for the Propaganda Techniques Corpus

Contents

1. Tasks
2. Evaluation scripts
3. Data format
4. Tools
5. Citation 


Tasks
--------------------------------------------
The Propaganda Techniques Corpus (PTC) is a corpus of articles annotated 
with propagandistic techniques at fine-grained level. The list of 
techniques is in file data/propaganda-techniques-names.txt. 
PTC enables for the development of automatic models for propaganda techniques 
identification (multi-class setting, task FLC). Furthermore, a binary 
task consisting in determining whether a sentence contains any propaganda 
technique is provided (SLC). See the paper in section "Citation" for 
further details. 


Evaluation scripts
--------------------------------------------

-Task FLC (task-FLC_scorer.py)

The evaluation script computes a variant of the precision, recall, and F-measure 
which takes into account partial overlappings between fragments (see 
https://propaganda.qcri.org/nlp4if-shared-task/data/evaluation_task_FLC.pdf for 
more details). The script evaluates all techniques together as well as each one 
in isolation.

The scorer requires file data/propaganda-techniques-names.txt. 
Such file contains the list of techniques used for scoring. 
Adding and removing items from the list will affect the outcome of the scorer. 
The script can be run as follows:

python3 task-FLC_scorer.py -s [prediction_file] -r [gold_folder]

As an example, we provide a "prediction_file" data/submission-task-FLC.tsv 
and run it as follows:

===
$ python3 task-FLC_scorer.py -s data/submission-task-FLC.tsv -r data/FLC-sample-labels
2019-06-26 14:42:29,735 - INFO - Checking user submitted file data/submission-task-FLC.tsv against gold folder data/FLC-sample-labels
2019-06-26 14:42:29,736 - INFO - Scoring user submitted file data/submission-task-FLC.tsv against gold file data/FLC-sample-labels
2019-06-26 14:42:29,736 - INFO - Scoring the submission with precision and recall method
2019-06-26 14:42:29,736 - INFO - Precision=5.000000/5=1.000000	Recall=4.769231/5=0.953846
2019-06-26 14:42:29,736 - INFO - F1=0.976378
2019-06-26 14:42:29,736 - INFO - Appeal_to_Authority: P=0.000000 R=0.000000 F1=1.000000
2019-06-26 14:42:29,736 - INFO - Appeal_to_fear-prejudice: P=0.000000 R=0.000000 F1=1.000000
2019-06-26 14:42:29,736 - INFO - Bandwagon: P=0.000000 R=0.000000 F1=1.000000
2019-06-26 14:42:29,736 - INFO - Black-and-White_Fallacy: P=0.000000 R=0.000000 F1=1.000000
2019-06-26 14:42:29,737 - INFO - Causal_Oversimplification: P=0.000000 R=0.000000 F1=1.000000
2019-06-26 14:42:29,737 - INFO - Doubt: P=0.000000 R=0.000000 F1=1.000000
2019-06-26 14:42:29,737 - INFO - Exaggeration,Minimisation: P=1.000000 R=1.000000 F1=1.000000
2019-06-26 14:42:29,737 - INFO - Flag-Waving: P=0.000000 R=0.000000 F1=1.000000
2019-06-26 14:42:29,737 - INFO - Loaded_Language: P=1.000000 R=1.000000 F1=1.000000
2019-06-26 14:42:29,737 - INFO - Name_Calling,Labeling: P=1.000000 R=1.000000 F1=1.000000
2019-06-26 14:42:29,737 - INFO - Obfuscation,Intentional_Vagueness,Confusion: P=0.000000 R=0.000000 F1=1.000000
2019-06-26 14:42:29,737 - INFO - Red_Herring: P=1.000000 R=1.000000 F1=1.000000
2019-06-26 14:42:29,737 - INFO - Reductio_ad_hitlerum: P=0.000000 R=0.000000 F1=1.000000
2019-06-26 14:42:29,737 - INFO - Repetition: P=0.000000 R=0.000000 F1=1.000000
2019-06-26 14:42:29,737 - INFO - Slogans: P=0.000000 R=0.000000 F1=1.000000
2019-06-26 14:42:29,738 - INFO - Straw_Men: P=0.000000 R=0.000000 F1=1.000000
2019-06-26 14:42:29,738 - INFO - Thought-terminating_Cliches: P=0.000000 R=0.000000 F1=1.000000
2019-06-26 14:42:29,738 - INFO - Whataboutism: P=0.000000 R=0.000000 F1=1.000000
===

The scorer for the SLC task is task-SLC_scorer.py. It can be run as follows

python3 task-SLC_scorer.py -s [prediction_file] -r [gold_file]

For example:

$ python3 task-SLC_scorer.py  -s dev/dev.task-SLC.labels -r dev/dev.task-SLC.labels

$ python3 task-SLC_scorer.py -s data/submission-task-SLC.tsv -r data/article736757214.task-SLC.labels
2019-06-26 14:46:11,048 - INFO - Checking user submitted file data/submission-task-SLC.tsv
2019-06-26 14:46:11,048 - INFO - OK: submission file data/submission-task-SLC.tsv format is correct
2019-06-26 14:46:11,048 - INFO - Scoring user submitted file data/submission-task-SLC.tsv
2019-06-26 14:46:11,052 - INFO - 
Precision=1.000000
Recall=0.800000
F1=0.888889

Data format
--------------------------------------------

-Task FLC

The corpus includes one tab-separated file per article with the following 
format: 

id   technique    begin_offset     end_offset

where id is the identifier of the article, technique is one out of the 18
techniques, begin_offset is the character where the covered span begins 
(incl.) and end_offset is the character where the covered span ends (
excl.). An example of such file is data/article736757214.task-FLC.labels. 

-Task SLC

The corpus includes one tab-separated file per article with the following format:

article_id	sentence_id	label

where article_id and sentence_id are the identifiers of the article and the sentence 
(the first sentence has id 1) and label={propaganda/non-propaganda}

Tools
--------------------------------------------

-- print_spans.py highlights the annotations in an article.

python3 print_spans.py -s [annotations_file] -t [article_file] -l

The -l options prints also line numbers of the article

For example:

python3 print_spans.py -t data/article736757214.txt -s data/article736757214.task-FLC.labels

-- split-train.sh splits the training sets into two sets, train-train and train-dev. The two subsets could be used
to locally train a classifier on train-train and score its predictions on train-dev
The script has no arguments, all parameters are inside the script (notice especially the variable $DATA_DIR which must
point to the parent folder of the training set).

-- fix_spans.py extends print_spans.py by allowing to add an offset to some of the annotations. 
Specifically, all annotations starting at a position greater or equal than FIX_FROM_CHAR_INDEX are shifted by OFFSET, 
which could be positive (right shift) or negative (left shift). If OFFSET > 0 than a file SPANS_FILE.fix is created
with the new annotations. 

$ python3 fix_spans.py -h

  -t ARTICLE_FILE, --text-file ARTICLE_FILE
                	file with text document
  -s SPANS_FILE, --spans-file SPANS_FILE
                        file with spans to be highlighted. 
  -l, --add-line-numbers
                        Prepend line numbers and index of the first character of the line on output.
  -f FIX_FROM_CHAR_INDEX, --fix-from-char-index FIX_FROM_CHAR_INDEX
                        Starting char index from which annotations are shifted by offset.
  -o OFFSET, --offset OFFSET
                        Offset by which annotations are shifted.

All annotations starting at a position greater or equal than FIX_FROM_CHAR_INDEX are shifted by OFFSET,
which could be positive (right shift) or negative (left shift). If OFFSET is not 0, than a file SPANS_FILE.fix is created
with the new annotations.

Example:
$ python3 fix_spans.py -s data/article736757214.task-FLC.labels -t data/article736757214.txt -l -f 150 -o -1
would shift 1 char to the left all annotations from starting from the 5th row (starting at the 169th char of the article)


-- check_corpus_annotations.py performs the following checks on the annotations of a number of articles:

1) Whether there is any annotation whose starting offset is beyond the lenght of the corresponding article (if option -f is used, deletes them)
2) Whether there is any annotation whose ending offset is beyond the length of the article (if option -f is used, they are trimmed such that they would end at the end of the article)
3) Whether there is any annotation that starts or ends with any of the following characters: " ", "\n", ".", ",", ":", ";" (if -f option is used, resize the spans such that they do not include such characters)

usage: check_corpus_annotations.py [-h] -t ARTICLES_FOLDER -s
                                   ANNOTATIONS_FOLDER [-o OUTPUT_FILE_SUFFIX]
                                   [-p PROPAGANDA_TECHNIQUES_FILE] [-d] [-b]
                                   [-c] [-f]

Performs some checks on a corpus. Example: check_corpus_annotations.py -s
data/article736757214.task-FLC.labels -t data/article736757214.txt

optional arguments:
  -h, --help            show this help message and exit
  -t ARTICLES_FOLDER, --articles-folder ARTICLES_FOLDER
                        path to the folder with articles
  -s ANNOTATIONS_FOLDER, --annotations-folder ANNOTATIONS_FOLDER
                        folders with annotation files.
  -o OUTPUT_FILE_SUFFIX, --output-file-suffix OUTPUT_FILE_SUFFIX
                        suffix of output files. Output file name will be
                        [annotations_folder]/article[article_id][suffix]
  -p PROPAGANDA_TECHNIQUES_FILE, --propaganda-techniques-list-file PROPAGANDA_TECHNIQUES_FILE
                        path to the file with the list of propaganda
                        techniques
  -d, --check-duplicated-annotations
                        Check for duplicated annotations in the same article.
  -b, --check-out-of-boundaries-annotations
                        Check for annotations that refer to spans out of the
                        boundaries of the article.
  -c, --check-start-end-chars-of-annotations
                        Check that annotations do not start/end with a
                        specific set of chars (see code).
  -f, --fix-errors      Fix errors whenever possible.

Example:

pipenv run python check_corpus_annotations.py -s data/articles_FLC_annotations -t data/articles -b -c -d




Classes overview
--------------------------------------------



Citation 
--------------------------------------------
Contact the organisers for instructions on what to cite when using the corpus
