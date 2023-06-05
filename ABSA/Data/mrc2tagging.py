#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  {
    "context": "Xinhua News Agency , Shanghai , August 31st , by reporter Jierong Zhou",
    "end_position": [
      2
    ],
    "entity_label": "ORG",
    "impossible": false,
    "qas_id": "0.2",
    "query": "organization entities are limited to companies, corporations, agencies, institutions and other groups of people.",
    "span_position": [
      "0;2"
    ],
    "start_position": [
      0
    ]
  }
"""

import os
import json


def mrc2bio(mrc_f):
    bio_f = []
    step = 3
    n_examples = int(len(mrc_f) / step)
    tagging_f = []
    for i_n in range(n_examples):
        mrcs = mrc_f[i_n * step: i_n * step + 3]
        words = mrcs[0]['context'].split(" ")
        labels = ["O"] * len(words)
        for mrc_one in mrcs:
            start = mrc_one['start_position']
            end = mrc_one['end_position']
            label_one = senttag2word[mrc_one['entity_label']]
            if len(start) != 0:
                for ip, pos in enumerate(start):
                    start_one = start[ip]
                    end_one = end[ip]
                    if start_one == end_one:
                        labels[start_one] = 'B-' + label_one
                    else:
                        labels[start_one] = 'B-' + label_one
                        for i in range(start_one+1, end_one+1):
                            labels[i] = 'I-' + label_one
        tagging_f.append((words, labels))
        for iw, w in enumerate(words):
            bio_f.append(words[iw] + " " + labels[iw] + '\n')
        bio_f.append('\n')
    return bio_f

if __name__ == "__main__":
    senttag2word = {'positive': 'POS', 'negative': 'NEG', 'neutral': 'NEU'}
    for data_type in ['Train', 'Test', "Dev"]:
        mrc_addr = os.path.join("./Data/laptop14", "{}.{}".format("mrc-uabsa", data_type.lower()))
        mrc_f = json.load(open(mrc_addr, encoding="utf-8"))
        bio_f = mrc2bio(mrc_f)
        save_file = os.path.join("./Data/laptop14", "{}.{}".format("tag-uabsa", data_type.lower()))

        with open(save_file, 'w') as writer:
            writer.writelines(bio_f)

    for data_type in ['Train', 'Test', "Dev"]:
        mrc_addr = os.path.join("./Data/res14", "{}.{}".format("mrc-uabsa", data_type.lower()))
        mrc_f = json.load(open(mrc_addr, encoding="utf-8"))
        bio_f = mrc2bio(mrc_f)
        save_file = os.path.join("./Data/res14", "{}.{}".format("tag-uabsa", data_type.lower()))

        with open(save_file, 'w') as writer:
            writer.writelines(bio_f)

    for data_type in ['res14', 'laptop14',]:
        mrc_addr = os.path.join("./Data/" + data_type, "{}.{}".format("mrc-adv", "test"))
        mrc_f = json.load(open(mrc_addr, encoding="utf-8"))
        bio_f = mrc2bio(mrc_f)
        save_file = os.path.join("./Data/" + data_type, "{}.{}".format("tag-adv", "test"))

        with open(save_file, 'w') as writer:
            writer.writelines(bio_f)