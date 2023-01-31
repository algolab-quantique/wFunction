from typing import Any, Union

from . import interpolate as terp
from .MPO_compress import MPO_compressing_sum
import jax.numpy as jnp
import numpy as np
import quimb.tensor as qtn
from numpy.polynomial.chebyshev import Chebyshev
from multimethod import multimethod, overload


def gen_Xi(i, domain, nbits):
	X = np.zeros((2, 2, 2, 2))
	X[0, :, 0, :] = np.eye(2)
	X[1, 1, 0, 1] = terp.bits2range(2**i, domain, nbits) - domain[0]
	X[1, :, 1, :] = np.eye(2)
	return X


def gen_X(domain: tuple[float, float], nbits: int):
	arr = [
		gen_Xi(i, domain, nbits).transpose(0, 2, 1, 3) for i in range(nbits - 1, -1, -1)
	]
	arr[0] = arr[0][1, :, :, :]
	arr[-1][1, 0, 0, 0] += domain[0]
	arr[-1][1, 0, 1, 1] += domain[0]
	arr[-1] = arr[-1][:, 0, :, :]
	return qtn.MatrixProductOperator(arr, site_tag_id="I{}")


def Chebyshevs(
	order: int,
	*,
	nqbits: int = 0,
	tol: float = 1e-13,
	X: Union[None, qtn.MatrixProductOperator] = None
) -> list[qtn.MatrixProductOperator]:
	"""Generate the chebyshev function evaluated with the operator X for arguement.
	If no operator is supplied, then a diagonal operator with equidistant eigenvalues
	in [-1,1] is used. The series of polynomial will not add to a convergent sum for
	any of the operator's eigenvalues outside the domain [-1,1]."""
	if X is None:
		assert nqbits != 0
		X = gen_X((-1, 1), nqbits)
	nqbits = X.L
	C0 = [np.array([[np.eye(2)]]) for i in range(nqbits)]
	C0[0] = C0[0][0, :, :, :]
	if nqbits > 1:
		C0[-1] = C0[-1][:, 0, :, :]
	C0 = qtn.MatrixProductOperator(C0, site_tag_id="I{}")
	C1 = X
	out = [C0, C1]
	for i in range(order - 1):
		tmp1 = 2 * X.apply(out[-1])
		tmp2 = -1 * out[-2]
		out.append(MPO_compressing_sum([tmp1, tmp2], tol, tol * 4))
		out[-1].site_tag_id = X.site_tag_id
	return out


def test_operator(mpo: qtn.MatrixProductOperator, do_print: bool = False):
	n = 2**mpo.L
	M = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			s = qtn.MPS_computational_state(format(i, ("0" + str(mpo.L) + "b")))
			l = qtn.MPS_computational_state(format(j, "0" + str(mpo.L) + "b"))
			s.site_ind_id = mpo.upper_ind_id
			l.site_ind_id = mpo.lower_ind_id
			M[i, j] = s @ (mpo | l)
	if do_print:
		print(M)
	return M


def test_diagonnal(mpo: qtn.MatrixProductOperator):
	n = 2**mpo.L
	M = np.zeros((n,))
	for i in range(n):
		s = qtn.MPS_computational_state(format(i, ("0" + str(mpo.L) + "b")))
		l = qtn.MPS_computational_state(format(i, "0" + str(mpo.L) + "b"))
		s.site_ind_id = mpo.upper_ind_id
		l.site_ind_id = mpo.lower_ind_id
		M[i] = s @ (mpo | l)
	return M


def func2MPO(fun, nqbit: int, tol: float):
	Cheb = auto_order_chebyshev(fun, tol)
	return Cheb2MPO(Cheb, nqbit, tol)


def Cheb2MPO(fun: Chebyshev, nqbit: int, tol: float):
	order = len(fun.coef) - 1
	MPO_poly = Chebyshevs(order, nqbits=nqbit, tol=tol)
	polysum = [coef * poly for poly, coef in zip(MPO_poly, fun)]
	mpo = MPO_compressing_sum(polysum, tol, 4 * tol)
	s2 = 1 / np.sqrt(2)
	# for i,t in enumerate(mpo):
	#     mpo[i] *=s2
	return mpo


@overload
def make_MPO(fun, nqbit: int, tol: float):
	return func2MPO(fun, nqbit, tol)


@overload
def make_MPO(fun: Chebyshev, nqbit: int, tol: float):
	return Cheb2MPO(fun, nqbit, tol)


@multimethod
def auto_order_chebyshev(f, tol: float):
	order = 1
	err = tol + 1
	while err > tol:
		Cf = Chebyshev.interpolate(f, order)
		err = max(abs(f(-1) - Cf(-1)), abs(f(1) - Cf(1)))
		order += max(int(np.floor(np.log(err / tol))), 1)
	Cf.trim(tol / 2)
	return Cf


@multimethod
def auto_order_chebyshev(f, g, tol: float):
	return auto_order_chebyshev(f, tol), auto_order_chebyshev(g, tol)


def to_lrud(
	tensor: qtn.Tensor,
	up: str,
	down: str,
	*,
	right_neighbour_inds=None,
	left_neighbour_inds=None
):
	if (right_neighbour_inds is None) == (left_neighbour_inds is None):
		raise ValueError(
			"either right_neighbour_inds or left_neighbour_inds must be defined"
		)
	inds = [*tensor.inds]
	inds.remove(up)
	inds.remove(down)
	if right_neighbour_inds is not None:
		for i in inds:
			if i in right_neighbour_inds:
				right = [i]
		inds.remove(right[0])
		left = inds
	else:
		for i in inds:
			if i in left_neighbour_inds:
				left = [i]
		inds.remove(left[0])
		right = inds
	return tensor.transpose(*left, *right, up, down).data


def pad_reshape(c):
	cshape = c.shape
	new_shape = (cshape[0], 1, *cshape[1:])
	return c.reshape(new_shape)


def reverse_qbit_order(mpo: qtn.MatrixProductOperator):
	L = len(mpo.tensors) - 1
	data = [
		mpo[-1].data,
		*[i.data.transpose(1, 0, 2, 3) for i in mpo[:].tensors[-2:0:-1]],
		mpo[0].data,
	]  # trickery to make sure the MPO is in order, because inverse listing doesn't work directly on the MPO.
	CMPO = qtn.MatrixProductOperator(
		data,
		site_tag_id=mpo.site_tag_id,
		upper_ind_id=mpo.upper_ind_id,
		lower_ind_id=mpo.lower_ind_id,
	)
	CMPO.permute_arrays("lrud")
	return CMPO


def cMPO_impl(
	Cf: Chebyshev,
	Cg: Chebyshev,
	chebyshev_order: int,
	nqbit: int,
	tol: float,
	endian: str,
) -> qtn.MatrixProductOperator:
	CMPO = Chebyshevs(chebyshev_order, nqbits=nqbit - 1, tol=tol)
	for i, c in enumerate(CMPO):
		CMPO[i].permute_arrays("lrud")
	if endian != "little":
		for i, c in enumerate(CMPO):  # reverse the MPO to account for desired endianess
			CMPO[i] = reverse_qbit_order(c)
	Y = np.array([[0.0, 1.0], [-1.0, 0.0]])
	I = np.array([[1.0, 0.0], [0.0, 1.0]])
	CMPOY = [
		coef
		* qtn.MatrixProductOperator(
			[
				*[mpo[i].data for i, x in enumerate(mpo.tensors[:-1])],
				pad_reshape(mpo[mpo.L - 1].data),
				Y.reshape(1, *Y.shape),
			],
			site_tag_id=mpo.site_tag_id,
			upper_ind_id=mpo.upper_ind_id,
			lower_ind_id=mpo.lower_ind_id,
		)
		for mpo, coef in zip(CMPO, Cf.coef)
	]
	CMPOI = [
		coef
		* qtn.MatrixProductOperator(
			[
				*[mpo[i].data for i, x in enumerate(mpo.tensors[:-1])],
				pad_reshape(mpo[mpo.L - 1].data),
				I.reshape(1, *I.shape),
			],
			site_tag_id=mpo.site_tag_id,
			upper_ind_id=mpo.upper_ind_id,
			lower_ind_id=mpo.lower_ind_id,
		)
		for mpo, coef in zip(CMPO, Cg.coef)
	]
	return MPO_compressing_sum([*CMPOY, *CMPOI], tol * 0.9, tol)


@multimethod
def controled_MPO(
	f, nqbit: int, tol: float, endian: str = "little"
) -> qtn.MatrixProductOperator:
	"""La valeur absolue de la fonction doit être inférieur à 1 sur le domaine (-1,1). nqbit inclue le qubit de control, le qubit de control est toujours le dernier, endian controle la convention d'encodage de la fonction."""
	g = lambda x: np.sqrt(1 - f(x) ** 2)
	Cf, Cg = auto_order_chebyshev(f, g, tol)
	chebyshev_order = max(len(Cf.coef), len(Cg.coef))
	return cMPO_impl(Cf, Cg, chebyshev_order, nqbit, tol, endian)


@multimethod
def controled_MPO(
	f, chebyshev_order: int, nqbit: int, tol: float, endian: str = "little"
) -> qtn.MatrixProductOperator:
	"""La valeur absolue de la fonction doit être inférieur à 1 sur le domaine (-1,1). nqbit inclue le qubit de control, le qubit de control est toujours le dernier, endian controle la convention d'encodage de la fonction."""
	g = lambda x: np.sqrt(1 - f(x) ** 2)
	Cg = Chebyshev.interpolate(g, chebyshev_order)
	Cf = Chebyshev.interpolate(f, chebyshev_order)
	return cMPO_impl(Cf, Cg, chebyshev_order, nqbit, tol, endian)


if __name__ == "__main__":
	import seaborn as sns

	sns.set_theme()
	import matplotlib.pyplot as plt

	def f(x):
		return jnp.exp(-(x**2) / 2)

	cheb_fun = Chebyshev.interpolate(f, 12, (-1, 1))
	nqbit = 10
	MPO_func = func2MPO(cheb_fun, nqbit, 1e-8)
	print(MPO_func)
	x = np.linspace(-1, 1, 2**nqbit)
	tfunc = test_diagonnal(MPO_func)
	plt.plot(x, f(x), label="f")
	plt.plot(x, tfunc, label="tfunc")
	plt.plot(x, cheb_fun(x), label="cheby")
	plt.legend()
	plt.show()
	plt.semilogy(x, abs(cheb_fun(x) - tfunc), label="tensor error")
	plt.show()
