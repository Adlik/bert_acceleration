import torch
import ctypes
import numpy as np

def load_clib():
	matmul_lib = ctypes.cdll.LoadLibrary("./libmm.so")
	matmul_lib.accelerate_fbgemm_matmul.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
	matmul_lib.accelerate_fbgemm_matmul.restype = ctypes.POINTER(ctypes.c_float)
	return matmul_lib

def get_shape_dim2(A, B):
	# Shape of matrixA is [M, K]
	# Shape of matrixB is [K, N]
	A_shape = A.shape
	B_shape = B.shape

	m = A_shape[0]
	k = A_shape[1]
	n = B_shape[1]

	return m, k, n

def get_shape_dim4(A, B):
	# Shape of matrixA is [batch_size, head_number, M, K]
	# Shape of matrixB is [batch_size, head_number, K, N]
	A_shape = A.shape
	B_shape = B.shape

	batch_size = A_shape[0]
	head_num = A_shape[1]

	m = A_shape[2]
	k = A_shape[-1]
	n = B_shape[-1]

	return batch_size, head_num, m, k, n

def Matmul_dim4(A, B):
	matmul_lib = load_clib()
	# Shape of matrixC is [batch_size, head_number, M, N]
	# Shape of matrixA is [batch_size, head_number, M, K]
	# Shape of matrixB is [batch_size, head_number, K, N]
	batch_size, head_num, m, k, n = get_shape_dim4(A, B)

	C = torch.rand(m, n)

	A = np.asarray(A, dtype=np.float32)
	B = np.asarray(B, dtype=np.float32)
	C = np.asarray(C, dtype=np.float32)
	C_Pointer = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

	C_list = []
	arrptr = ctypes.c_float * (m * n)

	for i in range(batch_size):
		for j in range(head_num):
			A_ = A[i][j]
			A_Pointer = A_.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
			B_ = B[i][j]
			B_Pointer = B_.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
			
			C = matmul_lib.accelerate_fbgemm_matmul(m, n, k, A_Pointer, B_Pointer, C_Pointer)

			addr = ctypes.addressof(C.contents)
			l = np.frombuffer(arrptr.from_address(addr), dtype=np.float32).tolist()
			C_list += l
			
	matmul_result_tensor = torch.tensor(C_list, dtype=torch.float32).contiguous().reshape((batch_size, head_num, m, n))
	return matmul_result_tensor

def Matmul_dim2(A, B):
	matmul_lib = load_clib()
	# Shape of matrixC is [M, N]
	# Shape of matrixA is [M, K]
	# Shape of matrixB is [K, N]
	m, k, n = get_shape_dim2(A, B)

	C = torch.rand(m, n)

	A = np.asarray(A, dtype=np.float32)
	B = np.asarray(B, dtype=np.float32)
	C = np.asarray(C, dtype=np.float32)
	C_Pointer = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

	C_list = []
	arrptr = ctypes.c_float * (m * n)

	A_Pointer = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
	B_Pointer = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
	
	C = matmul_lib.accelerate_fbgemm_matmul(m, n, k, A_Pointer, B_Pointer, C_Pointer)

	addr = ctypes.addressof(C.contents)
	l = np.frombuffer(arrptr.from_address(addr), dtype=np.float32).tolist()
	C_list += l
			
	matmul_result_tensor = torch.tensor(C_list, dtype=torch.float32).contiguous().reshape((m, n))
	return matmul_result_tensor

