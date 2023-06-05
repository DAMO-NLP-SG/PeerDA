import torch
from transformers.utils import logging


logger = logging.get_logger(__name__)

def tuple_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold pairs/triplets
    The input needs to be already processed
    """
    # number of true postive, gold standard, predicted aspect terms
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores


def generate_span(dataset, start_preds, end_preds, match_logits, label_mask, sample_ids, label_ids, match_labels, input_ids, flat=False):
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
    pred_batch = []
    gold_batch = []
    for batchid in batchid2sampleid:
        # gold
        sample_id, label_id = batchid2sampleid[batchid]
        gold_index = torch.where(match_labels[batchid] == True)
        for i in range(len(gold_index[0])):
            sub_text = dataset.tokenizer.decode((input_ids[batchid][gold_index[0][i]:gold_index[1][i]+ 1]))
            gold_batch.append((sample_id, label_id, sub_text))
        # pred
        if batchid  in batchid2spans:
            pred_spans = batchid2spans[batchid]
            for x in pred_spans:
                sub_text = dataset.tokenizer.decode(input_ids[batchid][x[0]:x[1]+ 1])
                pred_batch.append((sample_id, label_id, sub_text))
    return gold_batch, pred_batch
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


