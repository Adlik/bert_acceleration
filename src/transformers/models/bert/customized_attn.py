import torch
import math
from .quantized_matmul import quantize

def matmul_topk(A, B, k, mask, head_size):
    QA, _= quantize(A)
    QB, _= quantize(B)
    Qprod = torch.matmul(QA, QB)

    if mask is not None:
        Qprod = Qprod + mask * 10000

    _, idx = torch.topk(Qprod, k)

    prod = torch.matmul(A, B)
    val = prod.gather(dim=-1, index=idx)

    scores = torch.full_like(prod, 0.)
    scores.scatter_(src=val, index=idx, dim=-1)
    scores = scores / math.sqrt(head_size)
    scores[scores == 0.] = -10000.

    return scores

def matmul_local(A, B, r, mask, head_size):
    '''
    calculates matmul(A, B) only for diagonal elements and
    its 2r closest neighbors in a row
    '''
    if r >= B.shape[-1]: # k too large
        print("k too large")
        return torch.matmul(A, B)

    prod = torch.matmul(A, B)

    idx = torch.zeros(prod.shape[2], 2*r+1, device=A.device)
    idx = idx.long()
    for i in range(idx.shape[0]):
        for j in range(2*r+1):
            k = i + j - r
            if k > 0 and k < prod.shape[3]:
                idx[i][j] = k
            else:
                idx[i][j] = i
    
    idx = idx.expand(prod.shape[0], prod.shape[1], prod.shape[2], 2*r+1)
    
    val = prod.gather(dim=-1, index=idx)

    scores = torch.full_like(prod, 0.)
    scores.scatter_(src=val, index=idx, dim=-1)
    
    scores = scores / math.sqrt(head_size)

    scores[scores == 0.] = -10000.

    return scores

def matmul_customized(A, B, idx, mask, head_size):    
    prod = torch.matmul(A, B)
    val = prod.gather(dim=-1, index=idx)

    scores = torch.full_like(prod, 0.)
    scores.scatter_(src=val, index=idx, dim=-1)
    scores = scores / math.sqrt(head_size)
    scores[scores == 0.] = -10000.

    return scores

def matmul_customized_with_topk(A, B, k, idx2, mask, head_size):
    QA, _= quantize(A)
    QB, _= quantize(B)
    Qprod = torch.matmul(QA, QB)

    if mask is not None:
        Qprod = Qprod + mask * 10000

    _, idx1 = torch.topk(Qprod, k)

    idx = torch.cat((idx1, idx2), dim=-1)
    # idx = idx1
    
    prod = torch.matmul(A, B)
    val = prod.gather(dim=-1, index=idx)

    scores = torch.full_like(prod, 0.)
    scores.scatter_(src=val, index=idx, dim=-1)
    scores = scores / math.sqrt(head_size)
    scores[scores == 0.] = -10000.

    return scores

def idx_sliding_window(row, r, d, device):
    idx = torch.zeros(row, 2*r+1, device=device)
    idx = idx.long()
    for i in range(idx.shape[0]):
        for j in range(2*r+1):
            k = i + (j - r) * (d + 1)
            if k > 0 and k < 128:
                idx[i][j] = k
            else:
                idx[i][j] = i
    idx = idx.expand(1, 12, row, 2*r+1)
    return idx

def idx_block_local(row, r, device):
    idx = torch.zeros(row, r, device=device)
    idx = idx.long()
    for i in range(idx.shape[0]):
        for j in range(r):
            idx[i][j] = (i // r) * r + j
            if idx[i][j] > 127:
                idx[i][j] = 127
    idx = idx.expand(1, 12, row, r)
    return idx