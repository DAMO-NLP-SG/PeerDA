


import collections
import json
import math
import re
import string
import json
import torch
from transformers.models.bert import BasicTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)

def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or c == '\u200b' or c == '\xa0' or ord(c) == 0x202F:
        return True
    return False


def generate_span(dataset, start_preds, end_preds, match_logits, sample_ids, label_ids, label_mask, match_labels, input_ids, flat=False):
    """
    Compute span f1 according to query-based model output
    Args:
        start_preds: [bsz, seq_len]
        end_preds: [bsz, seq_len]
        match_logits: [bsz, seq_len, seq_len]
        label_mask: [bsz, seq_len]
        match_labels: [bsz, seq_len, seq_len]
        flat: if True, decode as flat-ner
    Returns:
        span-f1 counts, tensor of shape [3]: tp, fp, fn
    """
    start_label_mask = label_mask.bool()
    end_label_mask = label_mask.bool()
    match_labels = match_labels.bool()
    bsz, seq_len = start_label_mask.size()
    # [bsz, seq_len, seq_len]
    match_preds = match_logits > 0
    # [bsz, seq_len]
    start_preds = start_preds.bool()
    # [bsz, seq_len]
    end_preds = end_preds.bool()

    match_preds = (match_preds
                   & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                   & end_preds.unsqueeze(1).expand(-1, seq_len, -1))

    match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len)
                        & end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less or equal to end
    match_preds = match_label_mask & match_preds
    spans = torch.nonzero(match_preds).tolist()
    batchid2sampleid = {i:(torch.squeeze(sample_ids).tolist()[i], torch.squeeze(label_ids).tolist()[i]) for i in range(bsz)}
    batchid2spans = {}
    for one in spans:
        id_one, start_one, end_one = one
        if id_one not in batchid2spans:
            batchid2spans[id_one] = []
        batchid2spans[id_one].append((start_one, end_one))
    span_batch = []
    for batchid in batchid2sampleid:
        # gold
        sample_id, label_id = batchid2sampleid[batchid]
        example = dataset.all_data[sample_id * dataset.label_size + label_id]
        # pred
        pred = []
        if batchid  in batchid2spans:
            pred_spans = batchid2spans[batchid]
            for x in pred_spans:
                sub_text = dataset.tokenizer.decode(input_ids[batchid][x[0]:x[1]+ 1])
                start = example['tok_to_char_index'][x[0]]
                if x[1] == len(example['tok_to_char_index']) - 1:
                    end = len(example['raw_text'])
                    end = end - 1
                else:
                    end = example['tok_to_char_index'][x[1]+1]
                    if _is_whitespace(example['raw_text'][end - 1]):
                        end = end - 1
                span_batch.append((sample_id, label_id, start, end))

    return span_batch

def collate_to_max_length_bert(batch):
    """
    adapted form https://github.com/ShannonAI/mrc-for-flat-nested-ner
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
            tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    output = []

    for field_idx in range(6):
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    pad_match_labels = torch.zeros([batch_size, max_length, max_length], dtype=torch.long)
    pad_negative_match_labels = torch.zeros([batch_size, max_length, max_length], dtype=torch.long)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][6]
        negative_data = batch[sample_idx][7]
        pad_match_labels[sample_idx, : data.shape[1], : data.shape[1]] = data
        pad_negative_match_labels[sample_idx, : negative_data.shape[1], : negative_data.shape[1]] = negative_data
    output.append(pad_match_labels)
    output.append(pad_negative_match_labels)
    output.append(torch.stack([x[-2] for x in batch]))
    output.append(torch.stack([x[-1] for x in batch]))
    return output


def collate_to_max_length_roberta(batch):
    """
    adapted form https://github.com/ShannonAI/mrc-for-flat-nested-ner
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
            tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    output = []

    for field_idx in range(6):
        if field_idx == 0:
            pad_output = torch.full([batch_size, max_length], 1, dtype=batch[0][field_idx].dtype)
        else:
            pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    pad_match_labels = torch.zeros([batch_size, max_length, max_length], dtype=torch.long)
    pad_negative_match_labels = torch.zeros([batch_size, max_length, max_length], dtype=torch.long)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][6]
        negative_data = batch[sample_idx][7]
        pad_match_labels[sample_idx, : data.shape[1], : data.shape[1]] = data
        pad_negative_match_labels[sample_idx, : negative_data.shape[1], : negative_data.shape[1]] = negative_data
    output.append(pad_match_labels)
    output.append(pad_negative_match_labels)
    output.append(torch.stack([x[-2] for x in batch]))
    output.append(torch.stack([x[-1] for x in batch]))
    return output
