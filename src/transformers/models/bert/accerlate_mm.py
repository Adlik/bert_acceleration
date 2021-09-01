import torch
import ctypes
import numpy as np

def Matmul_(A, B):
	matmul_lib = ctypes.cdll.LoadLibrary("/home/yxj/zsh/transformers/examples/text-classification/accerlate_mm/libmm.so")
	A_shape = A.shape
	B_shape = B.shape
	m = A_shape[2]
	k = A_shape[-1]
	n = B_shape[-1]

	C = torch.rand(m, n)
	loop = A_shape[0] * A_shape[1]

	A = np.asarray(A, dtype=np.float32)
	B = np.asarray(B, dtype=np.float32)
	C = np.asarray(C, dtype=np.float32)
	C_Pointer = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

	L = []
	for i in range(A_shape[0]):
		for j in range(A_shape[1]):
			A_ = A[i][j]
			A_Pointer = A_.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
			B_ = B[i][j]
			B_Pointer = B_.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
			
			matmul_lib.mm.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
			matmul_lib.mm.restype = ctypes.POINTER(ctypes.c_float)
			C = matmul_lib.mm(m, n, k, A_Pointer, B_Pointer, C_Pointer)

			arrptr = ctypes.c_float * (m * n)
			addr = ctypes.addressof(C.contents)
			L_ = np.frombuffer(arrptr.from_address(addr), dtype=np.float32).tolist()
			L += L_
			
	matmul_result_tensor = torch.tensor(L, dtype=torch.float32).contiguous().reshape((A_shape[0], A_shape[1], m, n))
	return matmul_result_tensor