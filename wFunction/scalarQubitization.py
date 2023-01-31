#%%


from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Instruction, ParameterVector
from numpy import pi
import numpy as np
from scipy.signal import convolve, correlate, fftconvolve
from scipy.optimize import minimize
import jax.numpy as jnp
from qiskit.extensions import UnitaryGate
from scipy.linalg import expm
from numba import jit
import numba
from numba.typed import List as numbaList
from numpy.typing import NDArray
from matplotlib import pyplot as plt

# %%


@jit(nopython=True)
def exp(phi):
	r = np.abs(phi)
	if r > 0:
		a = phi / r
	else:
		a = 1
	return np.array(
		[[np.cos(r), np.sin(r) * a], [-np.sin(r) * a, np.cos(r)]], dtype=np.complex128
	)


@jit(nopython=True)
def phi_SU2_func(t, phi):
	T = np.array([[np.exp(1j * t), 0], [0, np.exp(-1j * t)]])
	out = exp(phi[0])
	for p in phi[1:]:
		out = out @ T @ exp(p)
	return -1 * out


@jit(nopython=True)
def W(phi: NDArray, t: NDArray, g: NDArray, *args):
	"""fonction de couts"""
	out = 0.0
	for time, f in zip(t, g):
		A = phi_SU2_func(time, numbaList(phi))
		o = A[0, 0] - f
		out += np.real(o * np.conj(o))
	return out


def opt_qubitizet(function, domain, precision, max_layers):
	"""
	Inspiré par [ Efficient phase-factor evaluation in quantum signal processing, Dong et al.].
	Mais on essai de construire une fonction qui ne correspond pas au contraintes dans l'artcle,
	et on utilise la méthode hybridisé entre Haah+Ying comme point de départ pour l'algorithme variationnelle.
	"""
	# Calcule de phi initiaux: on réduit l'amplitude de la fonction cible par 0.95 pour améliorer la stabilité de l'algorithme.
	ga = get_angles(lambda x: 0.95 * function(x), domain, max_layers, precision)
	phi = [*ga]
	n_layer = len(phi) - 1
	t = np.linspace(
		0, pi, 10 * n_layer
	)  # we need atleast 2*n_layer to satisfy the underlying nyquist theorem
	f = ZeroPiDomain(function, domain)
	g = f(t)
	print(phi)
	N = len(phi)

	def Wr(prpi, *args):
		phi = prpi[:N] + 1j * prpi[N:]
		return W(numbaList(phi), *args)

	print(W(phi, t, g))
	prpi = np.concatenate([np.real(phi), np.imag(phi)])
	result = minimize(Wr, prpi, args=(t, g), method="L-BFGS-B")
	prpi = result.x
	phi = prpi[:N] + 1j * prpi[N:]
	print(W(phi, t, g))
	return phi, ga.pi_over_2_domain


def opt_qubitizetrealangles(function, domain, precision, max_layers):
	ga = get_angles(lambda x: 0.95 * function(x), domain, max_layers, precision)
	phi = [*ga]
	n_layer = len(phi) - 1
	if ga.pi_over_2_domain:
		t = np.linspace(
			0, pi / 2, 10 * n_layer
		)  # we need atleast 2*n_layer to satisfy the underlying nyquist theorem
		t2 = np.linspace(
			0, pi, 10 * n_layer
		)  # some fooling around due to pi_over_2_domain unplanned appearence.
		f = ZeroPiDomain(function, domain)
		g = f(t2)
	else:
		t = np.linspace(
			0, pi, 10 * n_layer
		)  # we need atleast 2*n_layer to satisfy the underlying nyquist theorem
		f = ZeroPiDomain(function, domain)
		g = f(t)

	def WW(phi, t, g, *args):
		return W(numbaList(phi), t, g, *args)

	print(WW(phi, t, g))
	result = minimize(WW, phi, args=(t, g), method="L-BFGS-B")
	print(WW(result.x, t, g))
	return result.x, ga.pi_over_2_domain


def cosineWalk(nqbits, pow=1, max_angle=pi) -> QuantumCircuit:
	assert nqbits > 1
	x_qreg = QuantumRegister(nqbits, "x")
	circ = QuantumCircuit(x_qreg)
	for i in range(nqbits - 1):
		circ.cry(pow * max_angle, nqbits - 1, nqbits - i - 2)
		max_angle *= 0.5
	return circ


def phaseWalk(nqbits, pow=1, max_angle=pi) -> QuantumCircuit:
	assert nqbits > 1
	x_qreg = QuantumRegister(nqbits, "x")
	circ = QuantumCircuit(x_qreg)
	for i in range(nqbits - 1):
		circ.cp(pow * max_angle, nqbits - 1, nqbits - i - 2)
		max_angle *= 0.5
	return circ


def ZeroPiDomain(function, domain):
	"""
	Transform the input function from its original domain to [0,pi]
	"""
	c, d = domain
	return lambda x: function(x * (d - c) / pi + c)


def truncate(G, precision):
	"""
	Remove the high frequency components smaller than precision
	"""
	R = np.sum(G * np.conj(G))
	n = 0
	O = np.array([G[0]])
	r = np.sum(O * np.conj(O))
	while np.abs((r - R)) > precision**2:
		O = np.concatenate([G[: n + 1], G[-n:]])
		r = np.sum(O * np.conj(O))
		n += 1
	O[np.abs(O) < 0.1 * precision] = 0.0
	return O


def fft(a, n=None, axis=-1, norm="ortho"):
	"""physicist's convention"""
	return np.fft.fft(a, n, axis, norm)


def ifft(a, n=None, axis=-1, norm="ortho"):
	"""physicist's convention"""
	return np.fft.ifft(a, n, axis, norm)


def cpt_complement(fastfourierseries):
	"""Calcul le complément de la fonction donnée sous la forme d'une série de Fourier.
	Plus la fonction est proche de 1 n'importe ou sur le domaine, plus l'alogirhtme est instable.
	L'algorithme opère à précision fix de 64bits par nombre réel, à vous de vous assuré que c'est suffisant."""
	# D'abord, on réordonne la série de fourier en fréquence croissante.
	# Suppose qu'à l'entré on a affaire à l'ordre conventionnel des FFT (0, positives, négatives).
	n = len(fastfourierseries) // 2
	polynomial = np.fft.fftshift(fastfourierseries)

	# Calcul le produit de la fonction réciproque avec son conjugué; la correlation dans l'espace de fourier.
	# l'objectif est de calculé la représentation de Fourier de 1-|f(t)|**2
	r = -1 * correlate(polynomial, np.conj(polynomial))
	m = len(polynomial) % 2
	r[len(polynomial) - m] += 1

	# le polynome est réel, ses racines sont donc soit réel, ou en pair conjugué.
	# La méthode risque d'échoué si les racinnes réel ne sont pas d'ordre pair.
	# Cette echec ne peut pas se produire si la condition |f(t)| <= 1 est respecté.
	# Des précautions supplémentaires sont nécéssaire pour détecté et traiter les racinnes dégénéré.
	# A partir d'ici, il faudrait utilisé un arithmétique à point flotant avec un nombre de bit supérieur à 2Nlog2(N/e) ou N est le degrée du polynôme et e est la précision qu'on cherche à maintenir
	# Avec une précision modeste de 1e-4, et 10 racines ça donne 333 bit par nombre...
	roots = np.roots(r)
	oroots = roots.copy()
	m = len(roots)
	roots = roots[np.abs(roots) <= 1.0]
	n = len(roots)
	assert np.abs(n / m - 0.5) < 1e-15  # doit être 1/2
	# La fonction complémentaire est $w^{-\floor{n/2}}\prod_r (w-r)$ ou r sont les racines après filtrage.
	# À moins qu'il n'y ais que quelque racine, un calcule directe des coéfficient polynomial n'est pas possible numériquement.
	# Il y a typiquement des annulations catastrophique dû à la précision finie.
	# Effectué le produit des polynôme dans l'espace réciproque est plus stable.
	F = np.zeros(n + 1, dtype=np.complex128)
	F[0] = 1
	f = ifft(F, norm="forward")
	i = 0
	for root in roots:
		if i < n // 2:
			F[0] = 1
			F[-1] = -root
			F[1] = 0
		else:
			F[0] = -root
			F[1] = 1
			F[-1] = 0
		fr = ifft(F, len(f), norm="forward")
		f *= fr
		i += 1
	g = ifft(fastfourierseries, norm="forward")
	a = np.sqrt((1 - g * np.conj(g)) / (f * np.conj(f)))
	aa = np.average(a)
	f *= aa
	F = fft(f, norm="forward")
	sF = np.fft.fftshift(F)

	def W(sF):
		x = jnp.correlate(sF, jnp.conj(sF), mode="full") - r
		return jnp.sum(jnp.real(x * jnp.conj(x)))

	F = np.fft.ifftshift(sF)
	assert len(F) == len(fastfourierseries)
	return F


def cpt_Laurent_mats(function, domain, max_layers, precision):
	f = ZeroPiDomain(function, domain)
	t = np.linspace(0, pi, max_layers // 2 + 1)
	g = f(t)

	g = np.concatenate(
		[g[:-1], g[-1:0:-1]]
	)  # last element must be chopped for correct parity to be correct
	G = fft(
		g, norm="forward"
	)  # forward normalisation: we can truncate in Fourier's space without changing the normalisation.
	G = truncate(G, precision)

	rG = np.array([*reversed(G)])
	rG = np.concatenate([[rG[-1]], rG[:-1]])
	COMP = cpt_complement(G)
	rcomp = np.array([*reversed(COMP)])
	rcomp = np.concatenate([[rcomp[-1]], rcomp[:-1]])
	C = np.array([[G, COMP], [-np.conj(rcomp), np.conj(rG)]])
	n = len(rG) // 2
	assert (
		np.abs(np.linalg.det(C[:, :, n + 1])) < precision * 0.1
	), "The algorithm cannot reach target precision {}".format(precision)
	assert (
		np.abs(np.linalg.det(C[:, :, -n])) < precision * 0.1
	), "The algorithm cannot reach target precision {}".format(precision)
	return G, COMP


def Check_and_correct_ps(p: NDArray, s: NDArray) -> tuple[NDArray, NDArray, bool]:
	"""
	Check that p and s have the property that every other frequency coefficient is zero.
	If not, double their lenght by interspeccing zeros in the spectrum.
	"""
	lp = len(p)
	n = lp // 2
	assert lp > 0 and len(s) > 0 and len(s) == lp
	if (
		(np.abs(p[1:n:2]) < 5e-15).all()
		and (np.abs(p[n + 2 :: 2]) < 5e-15).all()
		and (np.abs(s[1:n:2]) < 5e-15).all()
		and (np.abs(s[n + 2 :: 2]) < 5e-15).all()
	):
		return p, s, False
	else:
		new_p = np.zeros(lp * 2 - 1, dtype=p.dtype)
		new_p[:lp:2] = p[: n + 1]
		new_p[lp::2] = p[n + 1 :]
		new_s = np.zeros(lp * 2 - 1, dtype=s.dtype)
		new_s[:lp:2] = s[: n + 1]
		new_s[lp::2] = s[n + 1 :]
		return new_p, new_s, True


def cpt_unitaries_transform(p, s, precision):
	"""
	Based on 'Product decomposition of Periodic Functions in Quantum Signal Processing, Jeongwan Haah Arxiv1806.10236v4'and 'Stable factorization for phase factors of quantum signal processing, Lexing Ying arxiv:2202.0261v3'
	"""
	p, s, doubled = Check_and_correct_ps(p, s)
	n = len(p) // 2
	U = []
	tp = np.fft.ifft([0, 1], n=len(p), norm="forward")
	ctd = np.conj(np.array([[tp, np.zeros(len(tp))], [np.zeros(len(tp)), np.conj(tp)]]))
	pt = np.fft.ifft(p, norm="forward")
	st = np.fft.ifft(s, norm="forward")
	e = np.sqrt((1 - np.abs(pt) ** 2) / np.abs(st) ** 2)
	for i in range(n, 0, -1):
		Cp = np.array([[p[i], s[i]], np.conj([-s[-i], p[-i]])])
		Cm = np.array([[p[-i], s[-i]], np.conj([-s[i], p[i]])])
		P = np.conj(Cp.T) @ Cp
		P /= np.trace(P)
		Q = np.conj(Cm.T) @ Cm
		Q /= np.trace(Q)
		Up, dp, Vp = np.linalg.svd(P)
		Um, dm, Vm = np.linalg.svd(Q)
		keta = Vp[0, :]
		ketb = Vm[0, :]
		ovrl = np.conj(keta) @ ketb
		assert np.abs(ovrl) < precision  # asserting orthonality
		ketb -= keta * ovrl  # enforcing orthogonality. error of order ovrl is present
		u = np.zeros((2, 2), dtype=np.complex128)
		u[0, :] = keta
		u[1, :] = ketb

		def upd_ptst():
			for l in range(len(pt)):
				pt[l], st[l] = [pt[l], st[l]] @ np.conj(u.T) @ (ctd[:, :, l]) @ u

		upd_ptst()
		p = np.fft.fft(pt, norm="forward")
		s = np.fft.fft(st, norm="forward")
		U.append(u)
	C = -np.array([[p[0], s[0]], np.conj([-s[0], p[0]])])
	C /= np.sqrt(0.5 * np.trace(np.conj(C.T) @ C))
	assert np.allclose(np.conj(C.T) @ C, np.eye(2), precision * 0.1, 1e-14)
	U.append(C)
	return U, doubled


def get_unitary_transform(function, domain, max_layers, precision=1e-4):
	p, s = cpt_Laurent_mats(function, domain, max_layers, precision)
	U, doubled = cpt_unitaries_transform(p, s, precision)
	return U, doubled


def var_get_rotation_matrices(
	function, domain, max_layers, precision=1e-4, var_method=opt_qubitizetrealangles
):
	"""
	Variationnal method for the computation of the rotation matrices, with the output formatted for scalar_qubitize.
	Rely on the direct method to compute initial point of the variationnal approach.
	Useful when the function to qubitize fall in the domain of instability of the direct method.
    
    Can optionnally optimize complex valued angles with 'var_method = opt_qubitizet'... no known use case.
	"""
	phi, pi_over2_domain = var_method(function, domain, precision, max_layers)
	return phi2rots(phi), pi_over2_domain


class get_rotations:
	def __init__(self, function, domain, max_layer, precision=1e-14):
		self.U, self.doubled = get_unitary_transform(
			function, domain, max_layer, precision
		)
		self.prev_matrix = np.eye(2)
		self.u_iter = iter(self.U)

	@property
	def pi_over_2_domain(self):
		"""return true of the domain is [0,pi/2], return false if it is [0,pi]"""
		return self.doubled

	def __len__(self):
		return len(self.U)

	def __iter__(self):
		return self

	def __next__(self):
		u = next(self.u_iter)
		mat = u @ self.prev_matrix
		self.prev_matrix = np.conj(u.T)
		return mat


def get_rotation_matrices(function, domain, max_layers, precision=1e-4):
	"""
	Direct method for the computation of the rotation matrices, with the output formatted for scalar_qubitize.
	"""
	gr = get_rotations(function, domain, max_layers, precision)
	return [*gr],gr.pi_over_2_domain


class get_angles:
	def __init__(self, function, domain, max_layers, precision=1e-4):
		self.had_pauli_Z = False
		self.Pz = np.array([[1, 0], [0, -1]])
		self.rot_iter = get_rotations(function, domain, max_layers, precision)
		self.__pi_over_2_domain = self.rot_iter.pi_over_2_domain

	@property
	def pi_over_2_domain(self):
		return self.__pi_over_2_domain

	def __iter__(self):
		return self

	def __next__(self):
		mat = np.conj(next(self.rot_iter).T)
		if self.had_pauli_Z:
			mat = self.Pz @ mat
		self.had_pauli_Z = np.abs(mat[0, 0] - mat[1, 1]) > 1e-10
		z = np.arccos(np.real(mat[0, 0]))
		if np.abs(mat[0, 1]) > 1e-10:  # avoid spurious pi/2 phase when the angle is 2pi
			z *= (1 - 2 * self.had_pauli_Z) * mat[0, 1] / np.abs(mat[0, 1])
		return z


def phi2rots_it(phis):
	for p in phis:
		yield expm(np.array([[0, (p)], [(-1 * np.conj(p)), 0]]))


def phi2rots(phis):
	return [*phi2rots_it(phis)]


class evalUmats_SU2_func:
	def __init__(self, param):
		assert param[0].shape == (2, 2)
		self.rots = param
		self.cpt_mats = False

	def __call__(self, t):
		T = np.array([[np.exp(1j * t), 0], [0, np.exp(-1j * t)]])
		out = self.rots[0]
		out = np.eye(2)
		for r in self.rots[:-1]:
			out = out @ np.conj(r.T) @ T @ r
		out = -1 * out @ np.conj(self.rots[-1].T)
		return out


class eval_SU2_func:
	def __init__(self, param):
		try:
			assert param[0].shape == (2, 2)
			self.rots = param
			self.cpt_mats = False
		except:
			self.phi = param
			self.cpt_mats = True

	def __call__(self, t):
		T = np.array([[np.exp(1j * t), 0], [0, np.exp(-1j * t)]])
		if self.cpt_mats == False:
			out = self.rots[0]
			for r in self.rots[1:]:
				out = out @ T @ np.conj(r.T)
			return out
		else:
			out = exp(self.phi[0])
			for p in self.phi[1:]:
				out = out @ T @ exp(p)
			return -1 * out#il faut trouvé la source du signe moins ici...


def qubitize_scalar(
	function,
	nqbits,
	domain,
	max_layers,
	precision=1e-4,
	cpt_rotations_matrices=var_get_rotation_matrices,
):
	"""
	pick a power of 2 for max_layer. something smaller than 2**nqbits. Actual number of layer of proposed solution is controlled by the precision.
	In principle, the qubitized function image must have a modulus equal or smaller than one, but the algorithm is unstable if the modulus is too close to one anywhere on the domain.
	This isn't a problem for smooth probability distribution.
	"""
	rots, pi_over_2_domain = cpt_rotations_matrices(
		function, domain, max_layers, precision
	)
	max_angle = pi
	if pi_over_2_domain:
		max_angle /= 2
	T = phaseWalk(nqbits, max_angle=max_angle)
	x_qreg = QuantumRegister(nqbits, "x")
	circ = QuantumCircuit(x_qreg)
	all_qubit = [*range(nqbits)]
    # out = self.rots[0]
	# 	for r in self.rots[1:]:
	# 		out = out @ T @ np.conj(r.T)
	# 	return out
	ugate = UnitaryGate(rots[0])
	circ.append(ugate,[nqbits-1])
	for u in rots[1:]:
		circ.append(T, all_qubit)
		ugate = UnitaryGate(u)
		circ.append(ugate, [nqbits - 1])
	return circ


# %%
