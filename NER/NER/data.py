import json
import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import logging
import random
import copy
import os
logger = logging.get_logger(__name__)
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}
class MRCNERDataset(Dataset):
    def __init__(self, json_path, tokenizer: None, max_length: int = 512, max_query_length = 64, possible_only=False, pad_to_maxlen=False, is_training=False, rate=1, DA=False, is_chinese=False, sizeonly=False, cache=False):
        if cache and DA and is_training and os.path.exists(json_path+".balanced-" + str(rate)):
            self.all_data = json.load(open(json_path+".balanced-" + str(rate), encoding="utf-8"))
        else:
            self.all_data = json.load(open(json_path, encoding="utf-8"))
            self.is_chinese = is_chinese
            self.rate = rate
            self.prompt()
            if DA and is_training:
                if sizeonly:
                    self.peering_size()
                else:
                    self.peering_category()
                self.balance()
                if cache:
                    with open(json_path+".balanced-" + str(rate), 'w') as writer:
                        json.dump(self.all_data, writer, ensure_ascii=False, sort_keys=True, indent=2)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_query_length = max_query_length
        self.possible_only = possible_only
        if self.possible_only:
            self.all_data = [
                x for x in self.all_data if x["start_position"]
            ]
        self.pad_to_maxlen = pad_to_maxlen

    def prompt(self):
        num_examples = len(self.all_data)
        num_impossible = len([1 for x in self.all_data if x["impossible"]])
        self.neg_ratio = (num_examples - num_impossible) / num_impossible
        new_datas = []
        all_spans = {}
        for data in self.all_data:
            qas_id = data['qas_id']
            context_id = qas_id.split(".")[0]
            label = data['entity_label']
            start_positions = data["start_position"]
            end_positions = data["end_position"]
            if context_id not in all_spans:
                all_spans[context_id] = {}

            all_spans[context_id][label] = (start_positions, end_positions)

        for data in self.all_data:
            qas_id = data['qas_id']
            context_id = qas_id.split(".")[0]
            label = data['entity_label']
            details = data['query']
            context = data['context']
            start_positions = data["start_position"]
            end_positions = data["end_position"]
            words = context.split()
            assert len(words) == len(context.split(" "))
            if self.is_chinese:
                query = '突出显示与“{}”相关的部分（如果有）。 含义：{}'.format(label, details)
            else:
                query = 'Highlight the parts (if any) related to "{}". Details: {}'.format(label, details)
            span_positions = {"{};{}".format(start_positions[i], end_positions[i]):" ".join(words[start_positions[i]: end_positions[i] + 1]) for i in range(len(start_positions))}
            all_spans_one = all_spans[context_id]
            negative_start_position = [all_spans_one[label_i][0] for label_i in all_spans_one if label_i != label]
            negative_end_position = [all_spans_one[label_i][1] for label_i in all_spans_one if label_i != label]
            negative_start_position = [y for x in negative_start_position for y in x]
            negative_end_position = [y for x in negative_end_position for y in x]
            new_data = {
                'context':words,
                'end_position':end_positions,
                'entity_label':label,
                'impossible':data['impossible'],
                'qas_id':data['qas_id'],
                'query':query,
                'span_position':span_positions,
                'start_position': start_positions,
                'negative_start_position': negative_start_position,
                "negative_end_position": negative_end_position,
            }
            new_datas.append(new_data)
        self.all_data = new_datas

    def peering_category(self):
        examples = self.all_data
        label_possible_count = {}
        label_dict = {}
        sample_dict = {}
        new_examples = []
        global_qas_id = 10000000
        for ie, example in enumerate(examples):
            label_id = example['entity_label']
            if label_id not in label_possible_count:
                label_possible_count[label_id] = 0

            if not example["impossible"]:
                answers = [example['span_position'][x] for x in example['span_position']]
                label_possible_count[label_id] += 1
            else:
                answers = [None]
            if label_id not in label_dict:
                label_dict[label_id] = []
            for answer in answers:
                label_dict[label_id].append((ie, answer))

        max_label_num = int(max([label_possible_count[x] for x in label_possible_count]) * (self.rate + 1))
        for label_id in label_dict:
            item = label_dict[label_id]
            gap = max_label_num - label_possible_count[label_id]
            if gap <= 0:
                continue
            possible_item = [x for x in item if x[1] is not None]
            possible_combinations = [(x,y) for x in possible_item for y in possible_item if x[0] != y[0]]
            possible_sample = random.sample(possible_combinations, min(gap, len(possible_combinations)))
            if self.neg_ratio < 1:
                impossible_combinations = [(x,y) for x in possible_item for y in item if y[1] is None and x != y]
                impossible_sample = random.sample(impossible_combinations, min(gap, len(impossible_combinations)))
            else:
                impossible_sample = []

            pair_data = []
            for pair in possible_sample + impossible_sample:
                seed = pair[0]
                target = pair[1]
                pair_data.append((seed[1], target[:1]))

            pair_data = list({}.fromkeys(pair_data).keys())
            sample_dict[label_id] = pair_data
            for pair in pair_data:
                seed = pair[0]
                target = pair[1]
                target_example = copy.copy(examples[target[0]])
                qas_id = target_example['qas_id'].split(".")[1]
                target_example['qas_id'] = str(global_qas_id) + "." + qas_id
                global_qas_id += 1
                if self.is_chinese:
                    target_example['query'] = '突出显示与“{}”类似的部分（如果有）。'.format(seed.replace(" ", ""))  #
                else:
                    target_example['query'] = 'Highlight the parts (if any) similar to: ' + seed  #
                new_examples.append(target_example)
        self.all_data = examples + new_examples

    def peering_size(self):
        examples = self.all_data
        label_possible_count = {}
        label_dict = {}
        sample_dict = {}
        new_examples = []
        global_qas_id = 10000000
        for ie, example in enumerate(examples):
            label_id = example['entity_label']
            if label_id not in label_possible_count:
                label_possible_count[label_id] = 0

            if not example["impossible"]:
                answers = [example['span_position'][x] for x in example['span_position']]
                label_possible_count[label_id] += 1
            else:
                answers = [None]
            if label_id not in label_dict:
                label_dict[label_id] = []
            for answer in answers:
                label_dict[label_id].append((ie, answer))

        for label_id in label_dict:
            item = label_dict[label_id]
            gap = int(label_possible_count[label_id] * (self.rate + 1))
            if gap <= 0:
                continue
            possible_item = [x for x in item if x[1] is not None]
            possible_combinations = [(x,y) for x in possible_item for y in possible_item if x[0] != y[0]]
            possible_sample = random.sample(possible_combinations, min(gap, len(possible_combinations)))
            if self.neg_ratio < 1:
                impossible_combinations = [(x,y) for x in possible_item for y in item if y[1] is None and x != y]
                impossible_sample = random.sample(impossible_combinations, min(gap, len(impossible_combinations)))
            else:
                impossible_sample = []

            pair_data = []
            for pair in possible_sample + impossible_sample:
                seed = pair[0]
                target = pair[1]
                pair_data.append((seed[1], target[:1]))

            pair_data = list({}.fromkeys(pair_data).keys())
            sample_dict[label_id] = pair_data
            for pair in pair_data:
                seed = pair[0]
                target = pair[1]
                target_example = copy.copy(examples[target[0]])
                qas_id = target_example['qas_id'].split(".")[1]
                target_example['qas_id'] = str(global_qas_id) + "." + qas_id
                global_qas_id += 1
                if self.is_chinese:
                    target_example['query'] = '突出显示与“{}”类似的部分（如果有）。'.format(seed.replace(" ", ""))  #
                else:
                    target_example['query'] = 'Highlight the parts (if any) similar to: ' + seed  #
                new_examples.append(target_example)
        self.all_data = examples + new_examples
    def balance(self):
        examples = self.all_data
        num_examples = len(examples)
        num_impossible = len([1 for x in examples if x["impossible"]])
        neg_keep_frac = (num_examples - num_impossible) / num_impossible
        neg_keep_mask = [x["impossible"] and random.random() < neg_keep_frac for x in examples]

        # keep all positive examples and subset of negative examples
        keep_mask = [(not examples[i]["impossible"]) or neg_keep_mask[i] for i in range(len(examples))]
        keep_indices = [i for i in range(len(keep_mask)) if keep_mask[i]]
        new_examples = [examples[i] for i in keep_indices]
        self.all_data = new_examples

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        """
        Args:
            item: int, idx
        Returns:
            tokens: tokens of query + context, [seq_len]
            token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
            start_labels: start labels of NER in tokens, [seq_len]
            end_labels: end labelsof NER in tokens, [seq_len]
            label_mask: label mask, 1 for counting into loss, 0 for ignoring. [seq_len]
            match_labels: match labels, [seq_len, seq_len]
            sample_idx: sample id
            label_idx: label id
        """
        data = self.all_data[item]
        tokenizer = self.tokenizer



        query = data["query"]
        context = data["context"]
        start_positions = data["start_position"]
        end_positions = data["end_position"]
        negative_start_positions = data["negative_start_position"]
        negative_end_positions = data["negative_end_position"]


        tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
        sequence_added_tokens = (
            tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
            if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
            else tokenizer.model_max_length - tokenizer.max_len_single_sentence
        )

        truncated_query = tokenizer.encode(
            query, add_special_tokens=False, truncation=True, max_length=self.max_query_length
        )

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(context):
            orig_to_tok_index.append(len(all_doc_tokens))
            if tokenizer.__class__.__name__ in [
                "RobertaTokenizer",
                "LongformerTokenizer",
                "BartTokenizer",
                "RobertaTokenizerFast",
                "LongformerTokenizerFast",
                "BartTokenizerFast",
            ]:
                sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
            elif tokenizer.__class__.__name__ in [
                'BertTokenizer'
            ]:
                sub_tokens = tokenizer.tokenize(token)
            elif tokenizer.__class__.__name__ in [
                'BertWordPieceTokenizer'
            ]:
                sub_tokens = tokenizer.encode(token, add_special_tokens=False).tokens
            else:
                sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)


        tok_start_positions = [orig_to_tok_index[x] for x in start_positions]
        tok_end_positions = []
        for x in end_positions:
            if x < len(context) - 1:
                tok_end_positions.append(orig_to_tok_index[x + 1] - 1)
            else:
                tok_end_positions.append(len(all_doc_tokens) - 1)

        tok_negative_start_positions = [orig_to_tok_index[x] for x in negative_start_positions]
        tok_negative_end_positions = []
        for x in negative_end_positions:
            if x < len(context) - 1:
                tok_negative_end_positions.append(orig_to_tok_index[x + 1] - 1)
            else:
                tok_negative_end_positions.append(len(all_doc_tokens) - 1)

        if self.pad_to_maxlen:
            truncation = TruncationStrategy.ONLY_SECOND.value
            padding_strategy = "max_length"
        else:
            truncation = TruncationStrategy.ONLY_SECOND.value
            padding_strategy = "do_not_pad"

        encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
            truncated_query,
            all_doc_tokens,
            truncation=truncation,
            padding=padding_strategy,
            max_length=self.max_length,
            return_overflowing_tokens=True,
            return_token_type_ids=True,
        )

        tokens = encoded_dict['input_ids']
        type_ids = encoded_dict['token_type_ids']
        attn_mask = encoded_dict['attention_mask']

        # find new start_positions/end_positions, considering
        # 1. we add query tokens at the beginning
        # 2. special tokens
        doc_offset = len(truncated_query) + sequence_added_tokens
        new_start_positions = [x + doc_offset for x in tok_start_positions if (x + doc_offset) < self.max_length - 1]
        new_end_positions = [x + doc_offset if (x + doc_offset) < self.max_length - 1 else self.max_length - 2 for x in
                             tok_end_positions]
        new_end_positions = new_end_positions[:len(new_start_positions)]

        new_negative_start_positions = [x + doc_offset for x in tok_negative_start_positions if (x + doc_offset) < self.max_length - 1]
        new_negative_end_positions = [x + doc_offset if (x + doc_offset) < self.max_length - 1 else self.max_length - 2 for x in
                             tok_negative_end_positions]
        new_negative_end_positions = new_negative_end_positions[:len(new_negative_start_positions)]

        label_mask = [0] * doc_offset + [1] * (len(tokens) - doc_offset - 1) + [0]



        assert all(label_mask[p] != 0 for p in new_start_positions)
        assert all(label_mask[p] != 0 for p in new_end_positions)
        assert all(label_mask[p] != 0 for p in new_negative_start_positions)
        assert all(label_mask[p] != 0 for p in new_negative_end_positions)

        assert len(label_mask) == len(tokens)
        start_labels = [(1 if idx in new_start_positions else 0)
                        for idx in range(len(tokens))]
        end_labels = [(1 if idx in new_end_positions else 0)
                      for idx in range(len(tokens))]

        seq_len = len(tokens)
        match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
        for start, end in zip(new_start_positions, new_end_positions):
            if start >= seq_len or end >= seq_len:
                continue
            match_labels[start, end] = 1

        negative_match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
        for start, end in zip(new_negative_start_positions, new_negative_end_positions):
            if start >= seq_len or end >= seq_len:
                continue
            negative_match_labels[start, end] = 1

        return [
            torch.LongTensor(tokens),
            torch.LongTensor(attn_mask),
            torch.LongTensor(type_ids),
            torch.LongTensor(start_labels),
            torch.LongTensor(end_labels),
            torch.LongTensor(label_mask),
            match_labels,
            negative_match_labels,
        ]