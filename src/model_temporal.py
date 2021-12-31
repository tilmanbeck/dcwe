import torch
from torch import nn, FloatTensor
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from transformers import BertModel, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

from data_helpers import isin

class TemporalClassificationModelNew(nn.Module):
    """"Class to train dynamic contextualized word embeddings for any classification task."""
    def __init__(self, nr_classes, vocab_filter, n_times=1, lambda_a=0.01, lm_model='bert-base-uncased'):
        """Initialize dynamic contextualized word embeddings model.

        Args:
            n_times: number of time points
            social_dim: dimensionality of social embeddings
            gnn: type of GNN (currently 'gat' and 'gcn' are possible)
        """

        super(TemporalClassificationModelNew, self).__init__()

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


class TemporalClassificationModel(nn.Module):
    """"Class to train dynamic contextualized word embeddings for any classification task."""
    def __init__(self, nr_classes, n_times=1, lm_model='bert-base-uncased'):
        """Initialize dynamic contextualized word embeddings model.

        Args:
            n_times: number of time points
            social_dim: dimensionality of social embeddings
            gnn: type of GNN (currently 'gat' and 'gcn' are possible)
        """

        super(TemporalClassificationModel, self).__init__()

        self.nr_classes = nr_classes

        self.bert = AutoModel.from_pretrained(lm_model)
        self.bert_emb_layer = self.bert.get_input_embeddings()
        self.offset_components = nn.ModuleList([OffsetComponent() for _ in range(n_times)])
        #self.social_components = nn.ModuleList([SocialComponent(social_dim, gnn) for _ in range(n_times)])
        self.linear_1 = nn.Linear(768, 100)
        self.dropout = nn.Dropout(0.2)
        self.linear_2 = nn.Linear(100, nr_classes)

    def forward(self, reviews, masks, segs, times, vocab_filter, embs_only=False):
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
        bert_embs = self.bert_emb_layer(reviews)
        offset_last = torch.cat(
            [self.offset_components[j](bert_embs[i]) for i, j in enumerate(F.relu(times - 1))],
            dim=0
        )
        offset_now = torch.cat(
            [self.offset_components[j](bert_embs[i]) for i, j in enumerate(times)],
            dim=0
        )
        offset_last = offset_last * isin(reviews, vocab_filter).float().unsqueeze(-1).expand(-1, -1, 768)
        offset_now = offset_now * isin(reviews, vocab_filter).float().unsqueeze(-1).expand(-1, -1, 768)

        # Compute dynamic type-level embeddings (input to contextualizing component)
        input_embs = bert_embs + offset_now

        # Only compute dynamic type-level embeddings (not fed into contextualizing component)
        if embs_only:
            return bert_embs, input_embs

        # Pass through contextualizing component
        output_bert = self.dropout(self.bert(inputs_embeds=input_embs, attention_mask=masks, token_type_ids=segs)[1])
        h = self.dropout(torch.tanh(self.linear_1(output_bert)))
        # we dont use the sigmoid function as we potentially deal with non-binary classification problems
        # output = torch.sigmoid(self.linear_2(h)).squeeze(-1)
        output = self.linear_2(h).view(-1, self.nr_classes)

        return offset_last, offset_now, output

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