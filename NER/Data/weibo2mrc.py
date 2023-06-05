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

def get_labels():
    """gets the list of labels for this data set."""
    return ["O", "B-GPE.NAM", "M-GPE.NAM", "E-GPE.NAM", "B-GPE.NOM", "E-GPE.NOM", \
            "B-LOC.NAM", "M-LOC.NAM", "E-LOC.NAM", "B-LOC.NOM", "M-LOC.NOM", "E-LOC.NOM", \
            "B-ORG.NAM", "M-ORG.NAM", "E-ORG.NAM", "B-ORG.NOM", "M-ORG.NOM", "E-ORG.NOM", \
            "B-PER.NAM", "M-PER.NAM", "E-PER.NAM", "B-PER.NOM", "M-PER.NOM", "E-PER.NOM", \
            "S-GPE.NAM", "S-LOC.NOM", "S-PER.NAM", "S-PER.NOM"]

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

def conll2mrc(conll_f):
    label_details = {"LOC": "山脉,河流自然景观的地点", "ORG": "组织包括公司,政府党派,学校,政府,新闻机构", "GPE":"按照国家,城市,州县划分的地理区域", "PER":"人名和虚构的人物形象"}
    mrc_f = []
    all_label_dict = []
    for ic, one in enumerate(conll_f):
        context = one[0]
        label_list = one[1]
        label_dict = {"LOC":[], "ORG":[], "GPE":[], "PER":[]}
        for il, label_one in enumerate(label_list):
            if label_one.startswith("S"):
                label_id = label_one.split(".")[0].split("-")[1]
                label_dict[label_id].append([il,il])
            elif label_one.startswith("B"):
                label_id_start = label_one.split(".")[0].split("-")[1]
                label_start = il
            elif label_one.startswith("E"):
                label_id_end = label_one.split(".")[0].split("-")[1]
                label_end = il
                try:
                    assert label_id_start == label_id_end
                    assert label_start < label_end
                    label_dict[label_id_end].append([label_start, label_end])
                except:
                    print('annotation error, ignored')
                label_id_start = 'bad'
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

if __name__ == "__main__":
    for data_type in ['train', 'test', 'dev']:
        data_file_path = os.path.join("./Data/weibo", "{}.{}".format("tag-ner", data_type))
        conll_f = read_conll(data_file_path)
        mrc_f = conll2mrc(conll_f)
        save_file = os.path.join("./Data/weibo", "{}.{}".format("mrc-ner", data_type))
        with open(save_file, 'w') as writer:
            json.dump(mrc_f,writer, ensure_ascii=False, sort_keys=True, indent=2)