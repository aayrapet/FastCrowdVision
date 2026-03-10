import torch
import torch.nn.functional as F
import pytest


# reference implementation (per image)
def HNM(classifications_reshaped, labels_reshaped, neg_pos_ratio=4):
    losses = F.cross_entropy(
        classifications_reshaped,
        labels_reshaped,
        reduction="none"
    )

    negative_indexes = torch.nonzero(labels_reshaped == 0, as_tuple=True)[0]
    positive_indexes = torch.nonzero(labels_reshaped > 0, as_tuple=True)[0]
    nb_positives = positive_indexes.numel()

    if nb_positives == 0:
        return positive_indexes

    _, indx = losses[negative_indexes].sort(descending=True)

    negative_indexes = negative_indexes[
        indx[:min(nb_positives * neg_pos_ratio, len(indx))]
    ]

    return torch.cat([positive_indexes, negative_indexes], dim=0)


# vectorized implementation
def HNMAX(classifications, labels, neg_pos_ratio=4):

    N, A, C = classifications.shape

    loss_c = F.cross_entropy(
        classifications.view(-1, C),
        labels.view(-1),
        reduction="none"
    ).view(N, A)

    pos = labels > 0
    loss_c[pos] = 0

    _, loss_idx = loss_c.sort(1, descending=True)
    _, idx_rank = loss_idx.sort(1)

    num_pos = pos.sum(1, keepdim=True)
    num_neg = torch.clamp(neg_pos_ratio * num_pos, max=A - 1)

    neg = idx_rank < num_neg.expand_as(idx_rank)

    selected = pos | neg

    return selected.view(-1).nonzero(as_tuple=True)[0]


# =========================================
# PYTEST
# =========================================

@pytest.mark.parametrize("N", [2, 4, 8, 20, 30])
def test_hnm_equivalence(N):

    torch.manual_seed(0)

    A = 8732
    C = 21

    classifications = torch.randn(N, A, C)
    labels = torch.zeros(N, A, dtype=torch.long)

    # ensure positives exist
    for i in range(N):
        pos_idx = torch.randperm(A)[:torch.randint(1, 20, (1,)).item()]
        labels[i, pos_idx] = torch.randint(1, C, (len(pos_idx),))

    # reference result
    gt_all = []
    for i in range(N):
        gt = HNM(classifications[i], labels[i])
        gt_all.append(gt + i * A)

    gt_all = torch.cat(gt_all).sort().values

    # vectorized result
    hnm_pc = HNMAX(classifications, labels).sort().values

    assert gt_all.numel() == hnm_pc.numel(), \
        f"Different number of anchors: {gt_all.numel()} vs {hnm_pc.numel()}"

    assert torch.equal(gt_all, hnm_pc)