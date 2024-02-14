"""
 Copyright (c) 2024 Alexandre Foley - Université de Sherbrooke

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program. If not, see <https://www.gnu.org/licenses/>.
 """

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
