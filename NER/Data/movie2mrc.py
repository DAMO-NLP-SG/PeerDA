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
            token_data_line, label_data_line  = token_label[1], token_label[0]
            cached_token.append(token_data_line)
            cached_label.append(label_data_line)
    return dataset_item_lst

def conll2mrc(conll_f):
    count = 0
    label_details = {'Actor': "a person who portrays a character in a performance.",
                     'Plot': "the sequence of events where each affects the next one through the principle of cause-and-effect.",
                     'Opinion': "a judgement, viewpoint, or statement that is not conclusive, rather than facts, which are true statements.",
                     'Award': "something given to a recipient as a token of recognition of excellence in a certain field.",
                     'Year': "time",
                     'Genre': "any form or type of communication in any mode (written, spoken, digital, artistic, etc.) with socially-agreed-upon conventions developed over time.",
                     'Origin': "The beginning of something.",
                     'Director': "A film director controls a film's artistic and dramatic aspects and visualizes the screenplay (or script) while guiding the film crew and actors in the fulfilment of that vision.",
                     'Soundtrack': "A soundtrack is recorded music accompanying and synchronised to the images of a motion picture, drama, book, television program, radio program, or video game.",
                     'Relationship': "a strong, deep, or close association or acquaintance between two or more people.",
                     'Character_Name': "a person or other being in a narrative.",
                     'Quote': 'a hypernym of quotation, as the repetition or copy of a prior statement or thought.'}
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
    for data_type in ['train', 'test']:
        data_file_path = os.path.join("./Data/movie", "{}.{}".format("tag-ner", data_type))
        conll_f = read_conll(data_file_path, delimiter="\t")
        mrc_f = conll2mrc(conll_f)
        save_file = os.path.join("./Data/movie", "{}.{}".format("mrc-ner", data_type))
        with open(save_file, 'w') as writer:
            json.dump(mrc_f,writer, ensure_ascii=False, sort_keys=True, indent=2)

# maxlen = 0
# for one in mrc_f:
#         maxlen += len(one['start_position'])
# label_dict = {}
# for one in dataset_item_lst:
#     for label in one[1]:
#         if label.startswith("B"):
#             label_name = label.replace("B-", "")
#             if label_name not in label_dict:
#                 label_dict[label_name] = 0
#             label_dict[label_name] += 1