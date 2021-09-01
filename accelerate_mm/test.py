import accelerate_mm
import time
import torch

def test_dim4(batch_size, head_num, m, k, n, iter):
    A = torch.randn(batch_size, head_num, m, k).cpu()
    B = torch.randn(batch_size, head_num, k, n).cpu()

    start_ref = time.time()
    for i in range(iter):
        C_ref = torch.matmul(A, B)
    time_ref = (time.time() - start_ref) / iter

    start_acc = time.time()
    for i in range(iter):
        C_acc = accelerate_mm.Matmul_dim4(A, B)
    time_acc = (time.time() - start_acc) / iter

    res = C_acc.equal(C_ref)
    accelerate_rate = (time_ref - time_acc) / time_ref
    print("Result is", res, "\naccelerate_rate is %f. "%(accelerate_rate))

test_dim4(4,4,128,64,128,10)