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


def read_conll(input_file, delimiter=" "):
    """load ner dataset from CoNLL-format files."""
    dataset_item_lst = []
    with open(input_file, "r", encoding="utf-8") as r_f:
        datalines = r_f.readlines()

    cached_token, cached_label = [], []
    for idx, data_line in enumerate(datalines):
        data_line = data_line.strip()
        if idx != 0 and len(data_line) == 0:
            dataset_item_lst.append([cached_token, cached_label])
            cached_token, cached_label = [], []
        else:
            token_label = data_line.split(delimiter)
            token_data_line, label_data_line = token_label[0], token_label[1]
            cached_token.append(token_data_line)
            cached_label.append(label_data_line)
    return dataset_item_lst

def wnut2mrc(conll_f):
    label_details = {
        "person": '''Names of people (e.g. Virginia Wade).''',
        "location": '''Names that are locations (e.g. France).''',
        "corporation": "Names of corporations (e.g. Google).",
        "product": '''Name of products (e.g. iPhone**).''',
        "creative work": "Names of creative works (e.g. Bohemian Rhapsody).",
        "group": "Names of groups (e.g. Nirvana, San Diego Padres).",
    }
    mrc_f = []
    all_label_dict = []
    label_count_count = {"person": 0, "location": 0, "corporation": 0, "product": 0, "creative work": 0, "group": 0}

    label_count_mrc = {"person": 0, "location": 0, "corporation": 0, "product": 0, "creative work": 0, "group": 0}
    for ic, one in enumerate(conll_f):
        context = one[0]
        label_list = one[1]
        label_dict = {"person":[], "location":[], "corporation":[], "product":[], "creative work":[], "group":[]}
        entity_flag = False
        for il, label_one in enumerate(label_list):
            label_one = label_one.replace("creative_work", "creative work")
            if label_one.startswith("B"):
                if entity_flag:
                    try:
                        if label_end == "bad":
                            label_dict[label_id_start].append([label_start, label_start])
                        else:
                            assert label_id_start == label_id_end
                            assert label_start < label_end
                            label_dict[label_id_start].append([label_start, label_end])
                        label_count_count[label_id_start] += 1
                    except:
                        print('annotation error, ignored')

                label_id_start = label_one.split(".")[0].split("-")[1]
                label_count_mrc[label_id_start] += 1
                label_start = il
                label_end = "bad"
                entity_flag = True
            elif label_one.startswith("I"):
                if entity_flag:
                    label_id_end = label_one.split(".")[0].split("-")[1]
                    label_end = il
            elif label_one.startswith("O"):
                if entity_flag:
                    try:
                        if label_end == "bad":
                            label_dict[label_id_start].append([label_start, label_start])
                        else:
                            assert label_id_start == label_id_end
                            assert label_start < label_end
                            label_dict[label_id_start].append([label_start, label_end])
                        label_count_count[label_id_start] += 1
                    except:
                        print('annotation error, ignored')
                    entity_flag = False
                    label_end = "bad"
                label_id_start = 'bad'
            if  entity_flag and il == len(label_list) - 1:
                try:
                    if label_end == "bad":
                        label_dict[label_id_start].append([label_start, label_start])
                    else:
                        assert label_id_start == label_id_end
                        assert label_start < label_end
                        label_dict[label_id_start].append([label_start, label_end])
                    label_count_count[label_id_start] += 1
                except:
                    print('annotation error, ignored')
                entity_flag = False

        all_label_dict.append(label_dict)
        for il, label_id in enumerate(label_details):
            entity_label = label_id
            query = label_details[label_id]
            qas_id = '{}.{}'.format(ic, il)
            label_item = label_dict[label_id]
            start_position = [x[0] for x in label_item]
            end_position = [x[1] for x in label_item]
            span_position = ['{};{}'.format(x[0], x[1]) for x in label_item]
            if span_position != []:
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
                "span_position": span_position,
                "start_position": start_position,
            }
            mrc_f.append(mrc_one)
        try:
            assert label_count_count == label_count_mrc
        except:
            print('get')
    return mrc_f

if __name__ == "__main__":
    for data_type in ['train', 'test', 'dev']:
        data_file_path = os.path.join("./Data/wnut", "{}.{}".format("mrc-tag",  data_type))
        conll_f = read_conll(data_file_path)
        mrc_f = wnut2mrc(conll_f)
        save_file = os.path.join("./Data/wnut", "{}.{}".format("mrc-ner", data_type))
        with open(save_file, 'w') as writer:
            json.dump(mrc_f,writer, ensure_ascii=False, sort_keys=True, indent=2)

