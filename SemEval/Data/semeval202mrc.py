import json
import os
# template_path = '../CCE/Data/CCE/mrc-cce.test'
# template = json.load(open(template_path, encoding="utf-8"))
from tqdm import tqdm

def load_data(data_folder, propaganda_techniques_file):
    file_list = os.listdir(data_folder)
    articles = {}
    for filename in sorted(file_list):
        with open(os.path.join(data_folder, filename), "r", encoding="utf-8") as f:
            articles[os.path.basename(filename).split(".")[0][7:]] = f.read()

    with open(propaganda_techniques_file, "r") as f:
        propaganda_techniques_names = [line.rstrip() for line in f.readlines()]

    return articles, propaganda_techniques_names

def read_gold(filename):
    gold_spans = {}
    with open(filename, "r") as f:
        for row in f.readlines():
            article_id, label, gold_span_start, gold_span_end = row.rstrip().split("\t")
            if article_id not in gold_spans:
                gold_spans[article_id] = []
            gold_spans[article_id].append((label, int(gold_span_start), int(gold_span_end)))
    return  gold_spans

def data2mrc(articles, gold_spans, propagandas):
    label_details = {'Appeal_to_Authority': "Stating that a claim is true simply because a valid authority or expert on the issue said it was true, without any other supporting evidence offered. We consider the special case in which the reference is not an authority or an expert in this technique, altough it is referred to as Testimonial in literature.",
     'Appeal_to_fear-prejudice': "Seeking to build support for an idea by instilling anxiety and/or panic in the population towards an alternative. In some cases, the support is built based on preconceived judgments.",
     'Bandwagon,Reductio_ad_hitlerum': "Persuading an audience to disapprove of an action or an idea by suggesting that the idea is popular with groups that are hated or in contempt by the target audience. It can refer to any person or concept with a negative connotation.",
     'Black-and-White_Fallacy': 'Presenting two alternative options as the only possibilities, when in fact more possibilities exist. As an the extreme case, tell the audience exactly what actions to take, eliminating any other possible choices (Dictatorship).',
     'Causal_Oversimplification': 'Assuming a single cause or reason when there are actually multiple causes for an issue. It includes transferring blame to one person or group of people without investigating the actual complexities of the issue.',
     'Doubt': 'Questioning the credibility of someone or something.',
     'Exaggeration,Minimisation': 'Either representing something in an excessive manner: making things larger, better, worse or making something seem less important or smaller than it really is.',
     'Flag-Waving': 'Playing on strong national feeling (or to any group; e.g., race, gender, political preference) to justify or promote an action or idea',
     'Loaded_Language': "Using specific words and phrases with strong emotional implications (either positive or negative) to influence an audience.",
     'Name_Calling,Labeling': "Labeling the object of the propaganda campaign as either something the target audience fears, hates, finds undesirable, or loves, praises.",
     'Repetition': "Repeating the same message over and over again so that the audience will eventually accept it.",
     'Slogans':  "A brief and striking phrase that may include labeling and stereotyping. Slogans tend to act as emotional appeals.",
     'Thought-terminating_Cliches': 'Words or phrases that discourage critical thought and meaningful discussion about a given topic. They are typically short, generic sentences that offer seemingly simple answers to complex questions or that distract attention away from other lines of thought.',
     'Whataboutism,Straw_Men,Red_Herring':  "A technique that attempts to discredit an opponent’s position by charging them with hypocrisy without directly disproving their argument.",
     }
    mrc_f = []
    global_exmaple_id = 0
    for article_id in tqdm(articles):
        article = articles[article_id]
        if article_id in gold_spans:
            gold_span_one = gold_spans[article_id]
        else:
            gold_span_one = []
        examples = []
        example_start = 0
        for i_c, ch in enumerate(article):
            example_end = i_c
            if ch == "\n":
                example_gold_spans = []
                for x in gold_span_one:
                    if x[1] >= example_start and x[2] - 1 <= example_end:
                        example_gold_spans.append((x[0], x[1], x[2] - 1))
                    elif  example_end >= x[1] >= example_start and x[2] - 1 > example_end:
                        example_gold_spans.append((x[0], x[1], example_end - 1))
                    elif  x[1] <= example_start and  example_start  <= x[2] - 1 <= example_end:
                        example_gold_spans.append((x[0], example_start,  x[2] - 1))
                text = article[example_start:example_end]
                if text != "":
                    example = {
                        'text':text,
                        "char_offset":example_start,
                        "spans":[(x[0],text[x[1] - example_start: x[2] - example_start + 1] , x[1] - example_start, x[2] - example_start) for x in example_gold_spans],
                    }
                    examples.append(example)
                example_start = i_c + 1
        for i_d, example in enumerate(examples):
            spans = example['spans']
            text = example['text']
            char_offset = example['char_offset']
            for i_l, entity_label in enumerate(label_details):
                qas_id = "{}.{}".format(global_exmaple_id, i_l)
                details = label_details[entity_label]
                impossible = True
                start_position = [x[2] for x in spans if x[0] == entity_label]
                end_position = [x[3] for x in spans if x[0] == entity_label]
                spans_text = [x[1] for x in spans if x[0] == entity_label]
                if len(spans_text) != 0:
                    impossible = False
                mrc_one = {
                    "article_id": article_id,
                    "context": text,
                    "end_position": end_position,
                    "entity_label": entity_label,
                    "impossible": impossible,
                    "qas_id": qas_id,
                    "query": details,
                    "spans": spans_text,
                    "start_position": start_position,
                    "char_offset":char_offset,
                }
                mrc_f.append(mrc_one)
            global_exmaple_id += 1
    return mrc_f
if __name__ == "__main__":
    propaganda_techniques_file = "./Data/semeval20/propaganda-techniques-names-semeval2020task11.txt"
    for data_type in ['train', 'test', 'dev']:
        data_file_path = os.path.join("./Data/semeval20", "{}-articles".format(data_type))
        label_file_path = os.path.join("./Data/semeval20", "{}-task-flc-tc.labels".format(data_type))
        articles, propagandas = load_data(data_file_path, propaganda_techniques_file)
        if data_type == "test":
            gold_spans = {}
        else:
            gold_spans = read_gold(label_file_path)
        mrc_f = data2mrc(articles, gold_spans, propagandas)
        save_file = os.path.join("./Data/semeval20", "{}.{}".format("mrc-semeval", data_type))
        with open(save_file, 'w') as writer:
            json.dump(mrc_f, writer, ensure_ascii=False, sort_keys=True, indent=2)