#!/usr/bin/python

import sys
import codecs

if len(sys.argv) < 3:
    sys.exit("Usage: %s <spans tsv file> <input text 1>" % (sys.argv[0]))

span_file = sys.argv[1]
file1 = sys.argv[2]

with open(span_file, "r") as f:
    spans = [line.rstrip().split("\t")[2:4] for line in f.readlines()]

#utf-8, utf-16, utf-32, utf-16-be, utf-16-le, utf-32-be, utf-32-le

with codecs.open(file1, "r", encoding="utf8") as f:
    s1 = f.read()
    print("File length: %d" % (len(s1)))
    for start, end in spans:
        selected_text = ""
        for i in range(int(start), int(end)):
            selected_text += s1[i]
        print("%s %s: -%s-" % (start, end, selected_text.replace("\n", " * ")))  # s1[int(start):int(end)]))

