import math
import torch
from torch import nn as nn
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel


def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]

    return token_h


class SpERT(BertPreTrainedModel):
    """ Span-based model to jointly extract entities and relations """

    VERSION = '1.1'

    def __init__(self, config: BertConfig, cls_token: int, relation_types: int, entity_types: int,
                 size_embedding: int, prop_drop: float, freeze_transformer: bool, max_pairs: int = 100):
        super(SpERT, self).__init__(config)

        # BERT model
        self.bert = BertModel(config)

        # layers
        self.entity_classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 3 + size_embedding, config.hidden_size * 3 + size_embedding),
            # nn.LeakyReLU(),
            nn.Dropout(prop_drop),
            nn.Linear(config.hidden_size * 3 + size_embedding, entity_types)
        )
        self.size_embeddings = nn.Embedding(100, size_embedding)
        self.dropout = nn.Dropout(prop_drop)

        self.gate_linear = nn.Linear(config.hidden_size, 1)
        self.binary_entity_classifier = nn.Linear(config.hidden_size, 1)

        self._cls_token = cls_token
        self._relation_types = relation_types
        self._entity_types = entity_types
        self._max_pairs = max_pairs

        # weight initialization
        self.init_weights()

        if freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                       entity_sizes: torch.tensor, mentions):
        # get contextualized token embeddings from last transformer layer
        context_masks = context_masks.float()
        h = self._encode_doc(encodings, context_masks)

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, b_entity_probs = self._classify_entities(
            encodings, h, entity_masks, size_embeddings, mentions)
        return entity_clf, b_entity_probs

    def _forward_inference(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                           entity_sizes: torch.tensor, mentions):
        # get contextualized token embeddings from last transformer layer
        context_masks = context_masks.float()
        # h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']
        h = self._encode_doc(encodings, context_masks)
        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, b_probs = self._classify_entities(
            encodings, h, entity_masks, size_embeddings, mentions)

        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)
        return entity_clf, b_probs

    def _encode_doc(self, encodings, context_masks):
        if encodings.shape[1] <= 512:
            h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']
        else:
            hs = []
            start_token, end_token = encodings[:, :1], encodings[:, -1:]
            start_mask, end_mask = context_masks[:, :1], context_masks[:, -1:]
            for i in range(math.ceil((encodings.shape[1]-2)/510)):
                cur_encoding = torch.cat([start_token, encodings[:, i*510+1: min(i*510+511, encodings.shape[1]-1)], end_token], dim=1)
                cur_context_mask = torch.cat([start_mask, context_masks[:, i*510+1: min(i*510+511, encodings.shape[1]-1)], end_mask], dim=1)
                cur_h = self.bert(input_ids=cur_encoding, attention_mask=cur_context_mask)['last_hidden_state']
                if i == 0:
                    hs.append(cur_h[:,:-1,:])
                elif i == math.ceil((encodings.shape[1]-2)/510) - 1:
                    hs.append(cur_h[:,1:,:])
                else:
                    hs.append(cur_h[:,1:-1,:])
            h = torch.cat(hs, dim=1)
        return h

    def _classify_entities(self, encodings, h, entity_masks, size_embeddings, mentions):
        # mentions are list of idx of spans that are the same mentions

        # max pool entity candidate spans
        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)
        # print('------->', m.shape, h.shape)
        # entity_spans_pool = m + h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
        entity_spans_pool = m + h.unsqueeze(1)
        entity_spans_pool = entity_spans_pool.max(dim=2)[0]

        entity_spans_pool, b_probs = self._update_entity_spans(entity_spans_pool, mentions)

        # get cls token as candidate context representation
        entity_ctx = get_token(h, encodings, self._cls_token)

        # create candidate representations including context, max pooled span and size embedding
        entity_repr = torch.cat([entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
                                 entity_spans_pool, size_embeddings], dim=2)
        entity_repr = self.dropout(entity_repr)

        # classify entity candidates
        entity_clf = self.entity_classifier(entity_repr)

        return entity_clf, b_probs

    def _update_entity_spans(self, entity_spans_pool, mentions):
        gates = torch.sigmoid(self.gate_linear(entity_spans_pool))
        b_probs = torch.sigmoid(self.binary_entity_classifier(entity_spans_pool))
        # b_probs = gates
        entity_spans_global = 1.0 * entity_spans_pool

        # below compute the max of multi-mentions in document
        # this serves as a key value dictionary and use max-pooling to fuse them
        for batch_idx, doc_mentions in enumerate(mentions):
            if len(doc_mentions) > 0:
                for mention in doc_mentions:
                    # cur_global_ele = torch.index_select(entity_spans_pool[batch_idx], 0, mention).max(dim=0)[0]
                    cur_global_ele = torch.index_select(entity_spans_global[batch_idx], 0, mention)
                    cur_probs = torch.index_select(b_probs[batch_idx], 0, mention)
                    if cur_probs.mean() > 0.1:
                        cur_global_ele = cur_probs * cur_global_ele
                        cur_global_ele = cur_global_ele.sum(dim=0) / cur_probs.sum()
                    
                        # cur_global_ele = torch.index_select(entity_spans_pool[batch_idx], 0, mention)
                        # Q = K = V = cur_global_ele.unsqueeze(1)
                        # cur_global_ele = self.attn(Q, K, V)[0].squeeze(1)
                        for m_idx, cur_men in enumerate(mention):
                            if cur_probs[m_idx] >= 0.3:
                                entity_spans_global[batch_idx, cur_men] = cur_global_ele
        # code end

        entity_spans_pool_new = torch.cat([entity_spans_pool * gates, (1 - gates) * entity_spans_global], dim=2)
        return entity_spans_pool_new, b_probs

    def forward(self, *args, inference=False, **kwargs):
        if not inference:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_inference(*args, **kwargs)


# Model access

_MODELS = {
    'spert': SpERT,
}


def get_model(name):
    return _MODELS[name]
