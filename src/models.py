from transformers import BertPreTrainedModel, BertModel, BertConfig

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

from data_helpers import isin

###################################
### TEMPORAL MODEL ### START ######
###################################

class TemporalClassificationModel(nn.Module):
    """"Class to train dynamic contextualized word embeddings for any classification task."""
    def __init__(self, nr_classes, vocab_filter, n_times=1, lambda_a=0.01, lm_model='bert-base-uncased'):
        """Initialize dynamic contextualized word embeddings model.

        Args:
            n_times: number of time points
            social_dim: dimensionality of social embeddings
            gnn: type of GNN (currently 'gat' and 'gcn' are possible)
        """

        super(TemporalClassificationModel, self).__init__()

        self.num_labels = nr_classes
        self.vocab_filter = vocab_filter
        self.lambda_a = lambda_a
        self.lambda_w = self.lambda_a / 0.001
        self.bert = AutoModel.from_pretrained(lm_model)
        self.bert_emb_layer = self.bert.get_input_embeddings()
        self.offset_components = nn.ModuleList([OffsetComponent() for _ in range(n_times)])
        #self.social_components = nn.ModuleList([SocialComponent(social_dim, gnn) for _ in range(n_times)])
        self.linear_1 = nn.Linear(768, 100)
        self.dropout = nn.Dropout(0.2)
        self.linear_2 = nn.Linear(100, nr_classes)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                timediff=None,
                labels=None,
                embs_only=False):
        """Perform forward pass.

        Args:
            reviews: tensor of tokenized reviews
            masks: tensor of attention masks
            segs: tensor of segment indices
            times: tensor of batch time points
            vocab_filter: tensor with word types for dynamic component
            embs_only: only compute dynamic type-level embeddings
        """

        # Retrieve BERT input embeddings
        bert_embs = self.bert_emb_layer(input_ids)
        offset_last = torch.cat(
            [self.offset_components[j](bert_embs[i]) for i, j in enumerate(F.relu(timediff - 1))],
            dim=0
        )
        offset_now = torch.cat(
            [self.offset_components[j](bert_embs[i]) for i, j in enumerate(timediff)],
            dim=0
        )
        offset_last = offset_last * isin(input_ids, self.vocab_filter).float().unsqueeze(-1).expand(-1, -1, 768)
        offset_now = offset_now * isin(input_ids, self.vocab_filter).float().unsqueeze(-1).expand(-1, -1, 768)

        # Compute dynamic type-level embeddings (input to contextualizing component)
        input_embs = bert_embs + offset_now

        # Only compute dynamic type-level embeddings (not fed into contextualizing component)
        if embs_only:
            return bert_embs, input_embs

        # Pass through contextualizing component
        output_bert = self.dropout(self.bert(inputs_embeds=input_embs, attention_mask=attention_mask, token_type_ids=token_type_ids)[1])
        h = self.dropout(torch.tanh(self.linear_1(output_bert)))
        # we dont use the sigmoid function as we potentially deal with non-binary classification problems
        # output = torch.sigmoid(self.linear_2(h)).squeeze(-1)
        output = self.linear_2(h).view(-1, self.num_labels)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # loss = loss_fct(output, labels.long().view(-1))
            loss = loss_fct(output.view(-1, self.num_labels), labels.view(-1))
            # print('Loss before offsetting: {:.5f}'.format(loss))
            loss += self.lambda_a * torch.norm(offset_now, dim=-1).pow(2).mean()
            # # print('Loss after offsetting: {:.5f}'.format(loss))
            loss += self.lambda_w * torch.norm(offset_now - offset_last, dim=-1).pow(2).mean()
            # # print('Loss after offsetting diff: {:.5f}'.format(loss))

        return SequenceClassifierOutput(
            loss=loss,
            logits=output,
        )


class OffsetComponent(nn.Module):
    """"Class implementing the dynamic component for social ablation."""

    def __init__(self):
        super(OffsetComponent, self).__init__()
        self.linear_1 = nn.Linear(768, 768)
        self.linear_2 = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)

    def forward(self, embs):
        h = self.dropout(torch.tanh(self.linear_1(embs)))
        offset = self.linear_2(h).unsqueeze(0)
        return offset

###################################
### TEMPORAL MODEL ### END ######
###################################


class GradientReversalFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambd):
        # store context for backprop
        ctx.save_for_backward(lambd)
        # forward pass is a no-op
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambd = ctx.saved_tensors[0]
        output = (grad_output * -lambd)

        return output, None

class TemporalPredictionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.decoder = nn.Linear(config.hidden_size, config.num_temporal_classes)
        self.lambd = torch.tensor(config.lambd, requires_grad=False) # lambda parameter for control of gradient reversal

    def forward(self, sequence_output):
        reversed_sequence_output = GradientReversalFn.apply(sequence_output, self.lambd)
        output = self.decoder(reversed_sequence_output)
        return output

class BertForSequenceClassificationAndDomainAdaptationConfig(BertConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lm_model = kwargs.get('lm_model')
        self.num_temporal_classes = kwargs.get('num_temporal_classes')
        self.lambd = kwargs.get('lambd')

class BertForSequenceClassificationAndDomainAdaptation(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_temporal_classes = config.num_temporal_classes
        self.config = config

        self.bert = BertModel.from_pretrained(config.lm_model)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        self.dropout = nn.Dropout(classifier_dropout)
        self.class_classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.time_classifier = TemporalPredictionHead(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
                time_labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        # very important detail here: we need to define here the name of the labels to be expected by the model such
        # that they are not removed from the initial datasets (they have to match as they respective data column names
        # are read from the model.forward signature

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
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        class_logits = self.class_classifier(pooled_output)
        time_logits = self.time_classifier(pooled_output)

        outputs = ([class_logits, time_logits], ) + outputs[2:]
        # outputs = ([class_logits], ) + outputs[2:]

        loss_fct = nn.CrossEntropyLoss()
        if labels is not None:
            loss = loss_fct(class_logits.view(-1, self.num_labels), labels.view(-1))
            if time_labels is not None and self.config.lambd > 0.0:
                time_loss = loss_fct(time_logits.view(-1, self.num_temporal_classes), time_labels.view(-1))
                loss += time_loss
            # print('Loss: {:.2f}, Label Class Loss: {:.2f}, Temporal Class Loss: {:.2f}'.format(loss, class_loss, time_loss))
            outputs = (loss, ) + outputs
            # loss_fct_per_sample = nn.CrossEntropyLoss(reduction='none')
            # outputs = (loss,
            #            loss_fct_per_sample(class_logits.view(-1, self.num_labels), class_labels.view(-1)),
            #            loss_fct_per_sample(topic_logits.view(-1, self.num_topic_labels), topic_labels.view(-1)),
            #            ) + outputs

        return outputs