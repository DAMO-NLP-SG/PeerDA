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


def read_line_examples_from_file(data_path):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(eval(tuples))
    print(f"Total examples = {len(sents)}")
    return sents, labels


def res2mrc(uabsa_file):
    mrc_f = []
    sents, labels = read_line_examples_from_file(uabsa_file)
    for ic, one_id in enumerate(sents):
        context = sents[ic]
        label = labels[ic]
        for il, label_id in enumerate(sentiment_details):
            entity_label = label_id
            query = sentiment_details[label_id]
            qas_id = '{}.{}'.format(ic, il)
            label_item = [(x[0], entity_label) for x in label if senttag2word[x[1]] == entity_label]
            start_position = [x[0][0] for x in label_item]
            end_position = [x[0][-1]  for x in label_item]
            category = [x[1] for x in label_item]
            if start_position != []:
                impossible = False
            else:
                impossible = True
            mrc_one = {
                "context": " ".join(context),
                "end_position": end_position,
                "entity_label": entity_label,
                "impossible": impossible,
                "qas_id": qas_id,
                "query": query,
                "category_span": category,
                "start_position": start_position,
            }
            mrc_f.append(mrc_one)
    return mrc_f

if __name__ == "__main__":
    uabsa = False
    if uabsa:
        sentiment_details = {
            "positive": "for aspect terms of positive sentiment.",
            "negative": "for aspect terms of negative sentiment.",
            "neutral": "for aspect terms of neutral sentiment.",
        }
        senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
    else:
        sentiment_details = {
            "aspect term": "detect aspect terms.",
        }
        senttag2word = {'POS': 'aspect term', 'NEG': 'aspect term', 'NEU': 'aspect term'}
    for data_type in ['Train', 'Test', "Dev"]:
        uabsa_file = os.path.join("./Data/res14", "{}.txt".format(data_type.lower()))
        mrc_f = res2mrc(uabsa_file)
        if uabsa:
            save_file = os.path.join("./Data/res14", "{}.{}".format("mrc-uabsa", data_type.lower()))
        else:
            save_file = os.path.join("./Data/res14", "{}.{}".format("mrc-ate", data_type.lower()))
        with open(save_file, 'w') as writer:
            json.dump(mrc_f,writer, ensure_ascii=False, sort_keys=True, indent=2)