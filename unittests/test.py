# test_hnm.py
import torch
import torch.nn.functional as F



#the function i coded which works for one image + for loop over images 
def HNM(classifications_reshaped, labels_reshaped, neg_pos_ratio=4):
    """
    classifications_reshaped: [A, C]
    labels_reshaped:          [A]
    returns:                  indices into A
    """
    losses = F.cross_entropy(
        classifications_reshaped,
        labels_reshaped,
        reduction="none"
    )

    negative_indexes = torch.nonzero(labels_reshaped == 0, as_tuple=True)[0]
    positive_indexes = torch.nonzero(labels_reshaped > 0, as_tuple=True)[0]
    nb_positives = positive_indexes.numel()

    if nb_positives == 0:
        return positive_indexes  # empty

    _, indx = losses[negative_indexes].sort(descending=True)
    negative_indexes = negative_indexes[
        indx[:min(nb_positives * neg_pos_ratio, len(indx))]
    ]

    return torch.cat([positive_indexes, negative_indexes], dim=0)

#Vectorized version max de groot , need to test their equivalence
def HNMAX(classifications, labels, neg_pos_ratio=4):
    """
    classifications: [N, A, C]
    labels:          [N, A]
    returns:         flat indices into (N*A)
    """

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


# ===============================
# UNIT TEST
# ===============================
def test_hnm_equivalence(
    N=5,
    A=8732,
    C=21,
    seed=0
):
    torch.manual_seed(seed)

    classifications = torch.randn(N, A, C)
    labels = torch.zeros(N, A, dtype=torch.long)

    # random positives (at least 1 per image)
    for i in range(N):
        pos_idx = torch.randperm(A)[:torch.randint(1, 20, (1,)).item()]
        labels[i, pos_idx] = torch.randint(1, C, (len(pos_idx),))

    # ---- ground truth (per image) ----
    gt_all = []
    for i in range(N):
        gt = HNM(classifications[i], labels[i])
        gt_all.append(gt + i * A)   # offset

    gt_all = torch.cat(gt_all).sort().values

    # ---- vectorized version ----
    hnm_pc = HNMAX(classifications, labels).sort().values

    # ---- assertions ----
    assert gt_all.numel() == hnm_pc.numel(), \
        f"Different number of selected anchors: {gt_all.numel()} vs {hnm_pc.numel()}"

    assert torch.all(gt_all == hnm_pc), \
        "HNMAX does NOT match reference HNM"

    print("✅ HNMAX matches reference HNM for N =", N)


#test runnin
if __name__ == "__main__":
    test_hnm_equivalence(N=2)
    test_hnm_equivalence(N=4)
    test_hnm_equivalence(N=8)
    test_hnm_equivalence(N=20)
    test_hnm_equivalence(N=30)