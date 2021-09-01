import torch

def quantize(A):
    '''
    quantize matrix A
    assert there are positive and negative values in matrix A
    '''
    # find the top and bottom clip_ratio number
    top_cut = A.max().item()
    bottom_cut = A.min().item()
    scale = max(top_cut, -bottom_cut)

    # map scale to INT8_MAX and -scale to -INT8_MAX
    QA = (A * 127 / scale).round()
    return QA, scale

def quantized_matmul_topk(A, B, k, mask):
    '''
    calculates matmul(Q(A), Q(B)), find its top K element
    then calculates the largest K elements of matmul(A, B)
    each row
    where Q(A) is matrix quantization
    '''
    if k >= B.shape[-1]: # k too large
        print("k too large")
        return torch.matmul(A, B)

    QA, _ = quantize(A)
    QB, _ = quantize(B)
    Qprod = torch.matmul(QA, QB)
    if mask is not None:
        Qprod = Qprod + mask

    _, idx = torch.topk(Qprod, k) # (try to) get the top k indices of each row
    
    prod = torch.matmul(A, B)
    val = prod.gather(index=idx, dim=-1) # mask prod by idx
    
    out = torch.full_like(prod, float('-inf'))
    out.scatter_(src=val, index=idx, dim=-1) # fills the values according to idx
    return out
