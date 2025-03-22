from abc import ABC
import torch
from torch import nn
import torch.nn.functional as F


class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass


class SpERTLoss(Loss):
    def __init__(self, rel_criterion, entity_criterion, model, optimizer, scheduler, max_grad_norm):
        self._rel_criterion = rel_criterion
        self._entity_criterion = entity_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm
        self.b_loss_f = nn.BCELoss()
        self.kl_loss_f = nn.KLDivLoss(log_target=True)

    def compute_kl_loss(self, entity_logits, b_entity_probs, mentions):
        entity_log_probs = F.log_softmax(entity_logits, dim=-1)
        prob1s = []
        prob2s = []
        for batch_idx, doc_mentions in enumerate(mentions):
            if len(doc_mentions) > 0:
                for mention in doc_mentions:
                    probs = []
                    cur_probs = torch.index_select(entity_log_probs[batch_idx], 0, mention)[:, 1:]
                    cur_b_probs = torch.index_select(b_entity_probs[batch_idx], 0, mention).squeeze()
                    cur_probs = torch.unbind(cur_probs, 0)
                    cur_b_probs = torch.unbind(cur_b_probs, 0)
                    for i in range(len(cur_probs)):
                        if cur_b_probs[i] > 0.5:
                            probs.append(cur_probs[i])
                    for i in range(len(probs) - 1):
                        for j in range(i, len(probs)):
                            prob1s.append(probs[i])
                            prob2s.append(probs[j])
        if len(prob1s):
            prob1 = torch.stack(prob1s)
            prob2 = torch.stack(prob2s)
            return (self.kl_loss_f(prob1, prob2) + self.kl_loss_f(prob2, prob1)) / 2
        else:
            return 0.0

    def compute(self, entity_logits, b_entity_probs, entity_types, entity_sample_masks, mentions):
        # kl loss
        kl_loss = self.compute_kl_loss(entity_logits, b_entity_probs, mentions)

        # entity loss
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])
        b_entity_probs = b_entity_probs.view(-1)
        entity_types = entity_types.view(-1)
        entity_sample_masks = entity_sample_masks.view(-1).float()

        b_entity_types = (entity_types > 0.5).float()
        entity_loss = self._entity_criterion(entity_logits, entity_types)
        entity_loss = (entity_loss * entity_sample_masks).sum() / entity_sample_masks.sum()
        b_entity_loss = self.b_loss_f(b_entity_probs, b_entity_types)
        # train_loss = entity_loss + b_entity_loss + kl_loss
        train_loss = entity_loss + b_entity_loss

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return train_loss.item()
