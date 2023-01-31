import quimb.tensor as qtn
from .MPO_compress import MPO_compressing_sum
import numpy as np


def pad_reshape(c):
	cshape = c.shape
	new_shape = (cshape[0], 1, *cshape[1:])
	return c.reshape(new_shape)


def cMPO(A: qtn.MatrixProductOperator, B: qtn.MatrixProductOperator, tol: float):
	A.permute_arrays()
	B.permute_arrays()
	Y = np.array([[0.0, 1.0], [-1.0, 0.0]])
	I = np.array([[1.0, 0.0], [0.0, 1.0]])
	CMPOY = [
		qtn.MatrixProductOperator(
			[
				*[A[i].data for i, x in enumerate(A.tensors[:-1])],
				pad_reshape(A[A.L - 1].data),
				Y.reshape(1, *Y.shape),
			],
			site_tag_id=A.site_tag_id,
			upper_ind_id=A.upper_ind_id,
			lower_ind_id=A.lower_ind_id,
		)
	]
	CMPOI = [
		qtn.MatrixProductOperator(
			[
				*[B[i].data for i, x in enumerate(B.tensors[:-1])],
				pad_reshape(B[B.L - 1].data),
				I.reshape(1, *I.shape),
			],
			site_tag_id=B.site_tag_id,
			upper_ind_id=B.upper_ind_id,
			lower_ind_id=B.lower_ind_id,
		)
	]
	return MPO_compressing_sum([*CMPOY, *CMPOI], tol * 0.9, tol)
