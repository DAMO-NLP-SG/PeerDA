from transformers.models.roberta.modeling_roberta import (
    RobertaModel,
    RobertaPreTrainedModel,
)
from transformers.models.bert.modeling_bert import (
    BertModel,
    BertPreTrainedModel,
)
from transformers.modeling_outputs import (
    ModelOutput
)
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
import torch
from torch import nn
from transformers.utils import logging
from torch.nn import BCEWithLogitsLoss
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
logger = logging.get_logger(__name__)

class BERT_MRC(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        weight_sum = 1 + 1 + config.weight_span
        self.weight_start = 1 / weight_sum
        self.weight_end = 1 / weight_sum
        self.weight_span = config.weight_span / weight_sum
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.span_transfer = MultiNonLinearProjection(config.hidden_size, config.hidden_size, config.hidden_dropout_prob,
                                                      intermediate_hidden_size=config.projection_intermediate_hidden_size)
        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_labels=None,
        end_labels=None,
        label_mask=None,
        match_labels=None,
        neagtive_match_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        batch_size, seq_len, hid_size = sequence_output.size()
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # improvement 1
        # adapted from https://github.com/ShannonAI/mrc-for-flat-nested-ner
        # for every position $i$ in sequence, should concate $j$ to
        # predict if $i$ and $j$ are start_pos and end_pos for an entity.
        # [batch, seq_len, hidden]
        span_intermediate = self.span_transfer(sequence_output)
        # [batch, seq_len, seq_len]
        span_logits = torch.matmul(span_intermediate, sequence_output.transpose(-1, -2))
        #
        total_loss = contrastive_loss = None
        if start_labels is not None and end_labels is not None:
            start_loss, end_loss, match_loss = self.compute_MRC_loss(start_logits, end_logits, span_logits,
                         start_labels, end_labels, match_labels, label_mask)
            total_loss = self.weight_start * start_loss + self.weight_end * end_loss + self.weight_span * match_loss
            # improvement 2
            if neagtive_match_labels is not None:
                contrastive_loss = self.compute_contrastive_loss(span_logits, match_labels, neagtive_match_labels, label_mask)
            if contrastive_loss is not None:
                total_loss += self.weight_span * 0.1 * contrastive_loss
            #
        if not return_dict:
            output = (start_logits, end_logits, span_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return UnifiedMRCOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            span_logits=span_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def compute_MRC_loss(self, start_logits, end_logits, span_logits,
                     start_labels, end_labels, match_labels, label_mask):
        batch_size, seq_len = start_logits.size()
        start_float_label_mask = label_mask.view(-1).float()
        end_float_label_mask = label_mask.view(-1).float()
        match_label_row_mask = label_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
        match_label_col_mask = label_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
        match_label_mask = match_label_row_mask & match_label_col_mask
        match_label_mask = torch.triu(match_label_mask, 0)  # start should be less equal to end

        float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        loss_fct = BCEWithLogitsLoss(reduction="none")
        start_loss = loss_fct(start_logits.view(-1), start_labels.view(-1).float())
        start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
        end_loss = loss_fct(end_logits.view(-1), end_labels.view(-1).float())
        end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
        match_loss = loss_fct(span_logits.view(batch_size, -1), match_labels.view(batch_size, -1).float())
        match_loss = match_loss * float_match_label_mask
        match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)

        return start_loss, end_loss, match_loss
    def compute_contrastive_loss(self,  span_logits, match_labels, neagtive_match_labels, label_mask):
        batch_size, seq_len, seq_len = span_logits.size()
        span_logits = span_logits.view(batch_size, -1)
        match_labels = match_labels.view(batch_size, -1)
        neagtive_match_labels = neagtive_match_labels.view(batch_size, -1)
        num_contrastive = torch.sum(match_labels, dim=1) * torch.sum(neagtive_match_labels, dim=1)
        if num_contrastive.sum() != 0:
            pos_logits = torch.sigmoid(span_logits) * match_labels + (1.0 - match_labels) * 10000.0
            neg_logits = torch.sigmoid(span_logits) * neagtive_match_labels + (1.0 - neagtive_match_labels) * -10000.0
            margin = 0
            diff = margin - (torch.min(pos_logits, dim=1)[0] - torch.max(neg_logits, dim=1)[0])
            contrastive_loss = diff.masked_fill(diff < 0, 0) * num_contrastive.bool().float()
            contrastive_loss = contrastive_loss.sum() / num_contrastive.bool().float().sum()
        else:
            contrastive_loss = None

        return contrastive_loss

class RoBERTa_MRC(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        weight_sum = 1 + 1 + config.weight_span
        self.weight_start = 1 / weight_sum
        self.weight_end = 1 / weight_sum
        self.weight_span = config.weight_span / weight_sum
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.span_transfer = MultiNonLinearProjection(config.hidden_size, config.hidden_size, config.hidden_dropout_prob,
                                                       intermediate_hidden_size=config.projection_intermediate_hidden_size)
        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_labels=None,
        end_labels=None,
        label_mask=None,
        match_labels=None,
        neagtive_match_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        batch_size, seq_len, hid_size = sequence_output.size()
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # improvement 1
        # adapted from https://github.com/ShannonAI/mrc-for-flat-nested-ner
        # for every position $i$ in sequence, should concate $j$ to
        # predict if $i$ and $j$ are start_pos and end_pos for an entity.
        # [batch, seq_len, hidden]
        span_intermediate = self.span_transfer(sequence_output)
        # [batch, seq_len, seq_len]
        span_logits = torch.matmul(span_intermediate, sequence_output.transpose(-1, -2))
        #
        total_loss = contrastive_loss = None
        if start_labels is not None and end_labels is not None:
            start_loss, end_loss, match_loss = self.compute_MRC_loss(start_logits, end_logits, span_logits,
                         start_labels, end_labels, match_labels, label_mask)
            total_loss = self.weight_start * start_loss + self.weight_end * end_loss + self.weight_span * match_loss
            # improvement 2
            if neagtive_match_labels is not None:
                contrastive_loss = self.compute_contrastive_loss(span_logits, match_labels, neagtive_match_labels, label_mask)
            if contrastive_loss is not None:
                total_loss += self.weight_span * 0.1 * contrastive_loss
        if not return_dict:
            output = (start_logits, end_logits, span_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return UnifiedMRCOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            span_logits=span_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def compute_MRC_loss(self, start_logits, end_logits, span_logits,
                     start_labels, end_labels, match_labels, label_mask):
        batch_size, seq_len = start_logits.size()
        start_float_label_mask = label_mask.view(-1).float()
        end_float_label_mask = label_mask.view(-1).float()
        match_label_row_mask = label_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
        match_label_col_mask = label_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
        match_label_mask = match_label_row_mask & match_label_col_mask
        match_label_mask = torch.triu(match_label_mask, 0)  # start should be less equal to end

        float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        loss_fct = BCEWithLogitsLoss(reduction="none")
        start_loss = loss_fct(start_logits.view(-1), start_labels.view(-1).float())
        start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
        end_loss = loss_fct(end_logits.view(-1), end_labels.view(-1).float())
        end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
        match_loss = loss_fct(span_logits.view(batch_size, -1), match_labels.view(batch_size, -1).float())
        match_loss = match_loss * float_match_label_mask
        match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)

        return start_loss, end_loss, match_loss
    def compute_contrastive_loss(self,  span_logits, match_labels, neagtive_match_labels, label_mask):
        batch_size, seq_len, seq_len = span_logits.size()
        span_logits = span_logits.view(batch_size, -1)
        match_labels = match_labels.view(batch_size, -1)
        neagtive_match_labels = neagtive_match_labels.view(batch_size, -1)
        num_contrastive = torch.sum(match_labels, dim=1) * torch.sum(neagtive_match_labels, dim=1)
        if num_contrastive.sum() != 0:
            pos_logits = torch.sigmoid(span_logits) * match_labels + (1.0 - match_labels) * 10000.0
            neg_logits = torch.sigmoid(span_logits) * neagtive_match_labels + (1.0 - neagtive_match_labels) * -10000.0
            margin = 0
            diff = margin - (torch.min(pos_logits, dim=1)[0] - torch.max(neg_logits, dim=1)[0])
            contrastive_loss = diff.masked_fill(diff < 0, 0) * num_contrastive.bool().float()
            contrastive_loss = contrastive_loss.sum() / num_contrastive.bool().float().sum()
        else:
            contrastive_loss = None

        return contrastive_loss



class MultiNonLinearProjection(nn.Module):
    'copy from https://github.com/ShannonAI/mrc-for-flat-nested-ner'
    def __init__(self, hidden_size, num_label, dropout_rate, act_func="gelu", intermediate_hidden_size=None):
        super(MultiNonLinearProjection, self).__init__()
        self.num_label = num_label
        self.intermediate_hidden_size = hidden_size if intermediate_hidden_size is None else intermediate_hidden_size
        self.classifier1 = nn.Linear(hidden_size, self.intermediate_hidden_size)
        self.classifier2 = nn.Linear(self.intermediate_hidden_size, self.num_label)
        self.dropout = nn.Dropout(dropout_rate)
        self.act_func = act_func

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        if self.act_func == "gelu":
            features_output1 = F.gelu(features_output1)
        elif self.act_func == "relu":
            features_output1 = F.relu(features_output1)
        elif self.act_func == "tanh":
            features_output1 = F.tanh(features_output1)
        else:
            raise ValueError
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2

@dataclass
class UnifiedMRCOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    start_loss: Optional[torch.FloatTensor] = None
    end_loss: Optional[torch.FloatTensor] = None
    match_loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    span_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]]= None