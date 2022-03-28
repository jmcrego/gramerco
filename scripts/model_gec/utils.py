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
    # torch_scatter.scatter(x, word_index, out=out, reduce=agregation)

    return out

    # max_len_word_seq = max(word_index.max().item() + 1, max_len)
    # bsz = x.size(0)
    # h = x.size(2)
    # L = x.size(1)
    #
    # _, c = torch.unique_consecutive(word_index, return_counts=True, dim=None)
    #
    # z = F.conv1d(
    #     word_index.view(bsz, 1, -1).float(),
    #     torch.tensor([-1, 1], device=word_index.device).float().view(1, 1, -1),
    #     padding="valid",
    # )
    # z = 1. - F.pad(z, (1, 1), "constant", 1)[:, :, :-1]
    # zz = z
    # for _ in range(c.max().item()):
    #     z = F.pad(
    #         (
    #             F.conv1d(
    #                 z.float(),
    #                 torch.tensor([1, 1], device=word_index.device).float().view(1, 1, -1),
    #                 padding="valid",
    #             )
    #             == 2
    #         ).long(),
    #         (1, 1),
    #         "constant",
    #         0,
    #     )[:, :, :-1]
    #     zz = torch.hstack([zz, z])
    # zr = zz.sum(-2)
    # idxs_y = (word_index * c.max() + zr).long()
    # e = torch.zeros(
    #     bsz,
    #     c.max().item() *
    #     max_len_word_seq,
    #     h,
    #     device=x.device,
    #     dtype=x.dtype)
    # idxs_x = torch.arange(bsz).unsqueeze(-1).expand(bsz, L).long()
    # e[idxs_x, idxs_y] = x
    # res = e.view(bsz, max_len_word_seq, c.max(), h)
    # if agregation == "sum":
    #     res = res.sum(-2)
    # elif agregation == "max":
    #     res, _ = res.max(-2)
    # return res


if __name__ == "__main__":

    torch.manual_seed(0)

    r = torch.rand(30).view(3, -1).cuda()
    x = torch.arange(60).float().view(3, 10, -1).cuda()
    word_index = r.cumsum(-1).long()
    print(word_index)
    print(x)
    res = word_collate(x, word_index, max_len=10)
    print(res)
