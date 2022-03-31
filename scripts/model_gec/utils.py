import torch
import torch.nn.functional as F
import torch_scatter


def word_collate(x, word_index, max_len=2, agregation="sum"):
    """Collate encodings of subwords tokens associated to the same word.

    Args:
        x (FloatTensor): tensor of dim Bsz x L x H
        word_index (LongTensor): tensor of ids of size Bsz x L
    Returns:
        FloatTensor: Agregation of the subword encoding acc. to the word
        indexing given through word_index.
    """
    if (word_index[:, -1] == 0).any():
        for i in range(len(word_index)):
            word_index[i, 1:][word_index[i, 1:] == 0] = word_index[i].max()
    out = torch.zeros_like(x, dtype=x.dtype, device=x.device)
    torch_scatter.segment_coo(x, word_index, out=out, reduce=agregation)

    return out


if __name__ == "__main__":

    torch.manual_seed(0)

    r = torch.rand(30).view(3, -1).cuda()
    x = torch.arange(60).float().view(3, 10, -1).cuda()
    word_index = r.cumsum(-1).long()
    print(word_index)
    print(x)
    res = word_collate(x, word_index, max_len=10)
    print(res)
