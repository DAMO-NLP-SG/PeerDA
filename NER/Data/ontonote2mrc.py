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
            token_data_line, label_data_line  = token_label[0], token_label[1]
            cached_token.append(token_data_line)
            cached_label.append(label_data_line)
    return dataset_item_lst

def conll2mrc(conll_f):
    count = 0
    label_details = {'ORG': "Companies, agencies, institutions, etc.",
                     'WORK_OF_ART': "Titles of books, songs, etc.",
                     'LOC': "Non-GPE locations, mountain ranges, bodies of water.",
                     'EVENT': "Named hurricanes, battles, wars, sports events, etc.",
                     'NORP': "Nationalities or religious or political groups",
                     'GPE': "Countries, cities, states.",
                     'PERSON': "People, including fictional.",
                     'FAC': "Buildings, airports, highways, bridges, etc",
                     'PRODUCT': "Vehicles, weapons, foods, etc.",
                     'LAW': "Named documents made into laws.",
                     'LANGUAGE': "Any named language."}
    mrc_f = []
    all_label_dict = []
    for ic, one in enumerate(conll_f):
        context = one[0]
        label_list = one[1]
        label_dict = {x:[] for x in label_details}
        last_label = None
        for il, label_one in enumerate(label_list):
            if last_label is not None and last_label.startswith("I") and not label_one.startswith("I"):
                assert label_id_start == label_id_end
                assert label_start < label_end
                label_dict[label_id_start].append([label_start, label_end])
            elif last_label is not None and last_label.startswith("B") and not label_one.startswith("I"):
                label_dict[label_id_start].append([label_start, label_start])
            if label_one.startswith("B"):
                count += 1
                label_id_start = label_one.split(".")[0].split("-")[1]
                label_start = il
                if il == len(label_list) - 1:
                    label_dict[label_id_start].append([il, il])
            elif label_one.startswith("I"):
                label_id_end = label_one.split(".")[0].split("-")[1]
                label_end = il
                if il == len(label_list) - 1:
                    assert label_id_start == label_id_end
                    assert label_start < label_end
                    label_dict[label_id_start].append([label_start, label_end])


            last_label = label_one
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
    return mrc_f

def get_fineclass(addr):
    with open(addr, 'r') as reader:
        ori_f = reader.readlines()
    label2entity = {}
    for idx, data_line in enumerate(ori_f):
        data_line = data_line.strip()
        if '\t' in data_line:
            data_list = data_line.split('\t')
            data_id, start, end, entity, label, entity_id = data_list
            if label not in label2entity:
                label2entity[label] = []
            label2entity[label].append(entity.lower().replace(" ", ""))
    for label in label2entity:
        item = label2entity[label]
        label2entity[label] = list(set(item))
    return  label2entity
if __name__ == "__main__":
    for data_type in ['dev', 'test', 'train']:
        data_file_path = os.path.join("./Data/ontonote5", "{}.{}".format("tag-ner", data_type))
        conll_f = read_conll(data_file_path, delimiter=" ")
        mrc_f = conll2mrc(conll_f)
        save_file = os.path.join("./Data/ontonote5", "{}.{}".format("mrc-ner", data_type))
        with open(save_file, 'w') as writer:
            json.dump(mrc_f,writer, ensure_ascii=False, sort_keys=True, indent=2)