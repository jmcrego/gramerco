import torch
import logging
import numpy as np


class CrossEntropyLoss(torch.nn.Module):
    """Loss compatible with GecBertModel.
    """
    def __init__(self, label_smoothing=0.0):
        super(CrossEntropyLoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, out, tgt, mask, x, mask_keep_prob=0):
        att_mask_out = out["attention_mask"][mask].bool()
        att_mask_in = x["tag_data"]["attention_mask"][mask].bool()

        y = out["tag_out"][mask][att_mask_out]
        t = tgt[mask][att_mask_in]

        mask_final = t.ne(0) | (
            torch.rand(
                t.shape,
                device=t.device) > mask_keep_prob)

        loss_tag = self.ce(y[mask_final], t[mask_final])

        return loss_tag


class DecisionLoss(torch.nn.Module):
    """Loss Compatible with GecBert2DecisionsModel.
    """
    def __init__(self, label_smoothing=0.0, beta=0.1):
        super(DecisionLoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.beta = beta
        logging.info("decision loss weight = " + str(self.beta))

    def forward(self, out, tgt, mask, x, mask_keep_prob=0):
        assert "decision_out" in out
        att_mask_out = out["attention_mask"][mask].bool()
        att_mask_in = x["tag_data"]["attention_mask"][mask].bool()

        y = out["decision_out"][mask][att_mask_out]
        t = tgt[mask][att_mask_in].bool().long()
        mask_final = t.ne(0) | (
            torch.rand(
                t.shape,
                device=t.device) > mask_keep_prob)
        loss_decision = self.ce(
            y[mask_final],
            t[mask_final],
        )
        # logging.debug(mask_final.long())
        error_mask = tgt[mask][att_mask_in].ne(0)
        y = out["tag_out"][mask][att_mask_out][error_mask]
        t = tgt[mask][att_mask_in][error_mask] - 1
        loss_tag = self.ce(
            y,
            t,
        )
        if not error_mask.any():
            loss_tag = torch.zeros_like(loss_tag)
        else:
            ...
        return loss_decision + self.beta * loss_tag


class CompensationLoss(torch.nn.Module):
    """Loss compatible with GecBertModel.
    """
    def __init__(self, label_smoothing=0.0):
        super(CompensationLoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, out, tgt, mask, x, mask_keep_prob=0):
        att_mask_out = out["attention_mask"][mask].bool()
        att_mask_in = x["tag_data"]["attention_mask"][mask].bool()

        y = out["tag_out"][mask][att_mask_out]
        t = tgt[mask][att_mask_in]
        y0 = y[:, 0]
        y1 = torch.logsumexp(y[:, 1:], -1) - torch.logsumexp(y, -1)
        decision = torch.stack((y0, y1), -1)
        loss_decision = self.ce(decision, tgt[mask][att_mask_in].bool().long())
        mask_final = t.ne(0) | (
            torch.rand(
                t.shape,
                device=t.device) > mask_keep_prob)
        loss_tag = self.ce(y[mask_final], t[mask_final])

        return loss_decision + loss_tag

class CETwoLoss(torch.nn.Module):
    """Loss compatible with GecBertVocModel.
    Performs cross entropy loss on the predictions of the last layers.
    """
    def __init__(self, label_smoothing=0.0, beta=1.):
        super(CETwoLoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.beta = beta
        logging.info("decision loss weight = " + str(self.beta))

    def forward(self, out, tgt, mask, x, tagger, mask_keep_prob=0):
        assert "voc_out" in out
        att_mask_out = out["attention_mask"][mask].bool()
        att_mask_in = x["tag_data"]["attention_mask"][mask].bool()

        y_tag = out["tag_out"][mask][att_mask_out]
        y_word = out["tag_out"][mask][att_mask_out]
        t = tgt[mask][att_mask_in].long()
        t_tag = tagger.id_to_tag_id_vec(t)
        t_word = tagger.id_to_word_id_vec(t)
        keep_mask = t_tag.ne(0) | (
            torch.rand(
                t.shape,
                device=t.device
            ) > mask_keep_prob
        )
        # logging.info("cpt inflects in tgt: " + str(((10 < t_tag) & (t_tag < 190)).long().sum().item()))
        loss_tag = self.ce(
            y_tag[keep_mask],
            t_tag[keep_mask],
        )

        voc_mask = t_word.ne(-1)
        y_voc = out["voc_out"][mask][att_mask_out][voc_mask]
        t_voc = t_word[voc_mask]
        loss_voc = self.ce(
            y_voc,
            t_voc,
        )
        if not voc_mask.any():
            loss_voc = torch.zeros_like(loss_voc)

        return loss_tag + self.beta * loss_voc

class CEThreeLoss(torch.nn.Module):
    """Loss compatible with GecBertInflVocModel.
    Performs cross entropy loss on the predictions of the last layers.
    """
    def __init__(self, label_smoothing=0.0, beta=1., gamma=1.):
        super(CEThreeLoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.beta = beta
        self.gamma = gamma
        logging.info("vocabulary loss weight = " + str(self.beta))
        logging.info("inflection loss weight = " + str(self.gamma))

    def forward(self, out, tgt, mask, x, tagger, mask_keep_prob=0):
        assert "voc_out" in out
        assert "infl_out" in out
        att_mask_out = out["attention_mask"][mask].bool()
        att_mask_in = x["tag_data"]["attention_mask"][mask].bool()

        y_tag = out["tag_out"][mask][att_mask_out]
        t = tgt[mask][att_mask_in].long()
        t_tag = tagger.id_to_tag_id_vec(t)
        # logging.info(str(torch.unique(t_tag, return_counts=True)))
        t_word = tagger.id_to_word_id_vec(t)
        t_infl = tagger.id_to_infl_id_vec(t)
        keep_mask = t_tag.ne(0) | (
            torch.rand(
                t.shape,
                device=t.device
            ) > mask_keep_prob
        )
        # logging.info("cpt inflects in tgt: " + str(((10 < t_tag) & (t_tag < 190)).long().sum().item()))
        # logging.info(">>>" + str(list(zip(*np.unique(t_tag[keep_mask].numpy(), return_counts=True)))))
        t_tag[t_tag.ne(0) & t_tag.ne(9)] = 1
        # logging.info("<<<" + str(list(zip(*np.unique(t_tag[keep_mask].numpy(), return_counts=True)))))
        loss_tag = self.ce(
            y_tag[keep_mask],
            t_tag[keep_mask],
        )

        infl_mask = t_infl.ne(-1)
        y_infl = out["voc_out"][mask][att_mask_out][infl_mask]
        t_infl = t_infl[infl_mask]
        loss_infl = self.ce(
            y_infl,
            t_infl,
        )

        voc_mask = t_word.ne(-1)
        y_voc = out["voc_out"][mask][att_mask_out][voc_mask]
        t_voc = t_word[voc_mask]
        loss_voc = self.ce(
            y_voc,
            t_voc,
        )
        if not infl_mask.any():
            loss_infl = torch.zeros_like(loss_infl)

        if not voc_mask.any():
            loss_voc = torch.zeros_like(loss_voc)

        return loss_tag + self.gamma * loss_infl + self.beta * loss_voc
