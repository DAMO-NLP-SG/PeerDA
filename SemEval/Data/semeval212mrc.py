#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
different from conll data format that annotates label on word, the annotation of SemEval is on character level.
Therefore, "span_position", 'start_position' and 'end_position' are character-level positions.

  {'context': 'Another violent and aggressive immigrant killing a innocent and intelligent US Citizen.... Sarcasm',
  'end_position': [4],
  'entity_label': 'Toxic',
  'impossible': False,
  'qas_id': '1.1',
  'query': 'posts that are rude, disrespectful, or unreasonable; and which can make users want to leave the conversation',
  'spans': ['violent and aggressive immigrant'],
  'start_position': [1]}
"""

import os
import json
import csv
import ast

def txt2mrc(input_file, delimiter=" "):
    """load ner dataset from csv files."""
    data = json.load(open(input_file, encoding="utf-8"))
    label_dict = {
        'Black-and-white Fallacy/Dictatorship': 'Presenting two alternative options as the only possibilities, when in fact more possibilities exist. As an the extreme case, tell the audience exactly what actions to take, eliminating any other possible choices (Dictatorship).',
        'Slogans': "A brief and striking phrase that may include labeling and stereotyping. Slogans tend to act as emotional appeals.",
        'Name calling/Labeling': "Labeling the object of the propaganda campaign as either something the target audience fears, hates, finds undesirable, or loves, praises.",
        'Loaded Language': ": Using specific words and phrases with strong emotional implications (either positive or negative) to influence an audience.",
        'Smears': "A smear is an effort to damage or call into question someone's reputation, by propounding negative propaganda. It can be applied to individuals or groups.",
        'Causal Oversimplification': 'Assuming a single cause or reason when there are actually multiple causes for an issue. It includes transferring blame to one person or group of people without investigating the actual complexities of the issue.',
        'Exaggeration/Minimisation': 'Either representing something in an excessive manner: making things larger, better, worse or making something seem less important or smaller than it really is.',
        'Appeal to fear/prejudice': "Seeking to build support for an idea by instilling anxiety and/or panic in the population towards an alternative. In some cases, the support is built based on preconceived judgments.",
        'Reductio ad hitlerum': "Persuading an audience to disapprove of an action or an idea by suggesting that the idea is popular with groups that are hated or in contempt by the target audience. It can refer to any person or concept with a negative connotation.",
        'Repetition': "Repeating the same message over and over again so that the audience will eventually accept it.",
        'Glittering generalities (Virtue)': "These are words or symbols in the value system of the target audience that produce a positive image when attached to a person or an issue.",
        "Misrepresentation of Someone's Position (Straw Man)": "When an opponent’s proposition is substituted with a similar one, which is then refuted in place of the original proposition.",
        'Doubt': 'Questioning the credibility of someone or something.',
        'Obfuscation, Intentional vagueness, Confusion': "Using words that are deliberately not clear, so that the audience can have their own interpretations.",
        'Whataboutism': "A technique that attempts to discredit an opponent’s position by charging them with hypocrisy without directly disproving their argument.",
        'Flag-waving': 'Playing on strong national feeling (or to any group; e.g., race, gender, political preference) to justify or promote an action or idea',
        'Thought-terminating cliché': 'Words or phrases that discourage critical thought and meaningful discussion about a given topic. They are typically short, generic sentences that offer seemingly simple answers to complex questions or that distract attention away from other lines of thought.',
        'Presenting Irrelevant Data (Red Herring)': "Introducing irrelevant material to the issue being discussed, so that everyone's attention is diverted away from the points made.",
        'Appeal to authority': "Stating that a claim is true simply because a valid authority or expert on the issue said it was true, without any other supporting evidence offered. We consider the special case in which the reference is not an authority or an expert in this technique, altough it is referred to as Testimonial in literature.",
        'Bandwagon': 'Attempting to persuade the target audience to join in and take the course of action because "everyone else is taking the same action."'}
    label_size = len(label_dict)
    mrc_f = []
    for i_d, data_one in enumerate(data):
        text = data_one['text']
        data_labels = data_one['labels']
        label_dict_one = {x:[] for x in label_dict}
        for i_l, data_label in enumerate(data_labels):
            start_position = data_label['start']
            end_position = data_label['end']
            label_name = data_label['technique']
            label_span = data_label['text_fragment']
            if label_name in label_dict:
                label_dict_one[label_name].append((start_position, end_position, label_span))
            else:
                print('label not find!')
        for i_l, label_name in enumerate(label_dict):
            entity_label = label_name
            qas_id = "{}.{}".format(i_d, i_l)
            details = label_dict[entity_label]
            impossible = True
            start_position = []
            end_position = []
            spans = []
            if label_dict_one[label_name] != []:
                impossible = False
                for item in label_dict_one[label_name]:
                    start_position.append(item[0])
                    if item[2] == text[item[0]:item[1]]:
                        end_position.append(item[1] - 1)
                    else:
                        end_position.append(item[1])
                    spans.append(item[2])

            mrc_one = {
                "context": text,
                "end_position": end_position,
                "entity_label": entity_label,
                "impossible": impossible,
                "qas_id": qas_id,
                "query": details,
                "spans": spans,
                "start_position": start_position,
            }
            mrc_f.append(mrc_one)
    return mrc_f

if __name__ == "__main__":
    for data_type in ['train', 'test', 'dev']:
        data_file_path = os.path.join("./Data/semeval21", "{}_set.{}".format(data_type, "txt"))
        mrc_f = txt2mrc(data_file_path)
        save_file = os.path.join("./Data/semeval21", "{}.{}".format("mrc-semeval", data_type))
        with open(save_file, 'w') as writer:
            json.dump(mrc_f,writer, ensure_ascii=False, sort_keys=True, indent=2)