from jax.config import config

config.update("jax_enable_x64", True)


from typing import Union
import quimb as qb
import quimb.tensor as qtn

# import quantit as qtt
import numpy as np
import jax.numpy as jnp
from quimb.tensor.optimize import TNOptimizer
from .compress_algs import layer_SVD_optimizer, MPS_compression
from .generate_simple_MPO import generate_id_MPO
import re

qtn.MatrixProductState
TN = qtn.TensorNetwork


def log4(x):
	return jnp.log(x) / jnp.log(4)


def link_pyramid_gate_num(bond_size: int, bond: int):
	if bond_size <= 1:
		return 0
	if bond % 2 == 0:
		return int(jnp.ceil((log4(bond_size / 2) if bond_size > 2 else 0) + 1))
	else:
		return int(jnp.ceil(log4(bond_size)))


def idx_list(site: int, gate_num: int):
	idx = "l{}q{}"
	l = [idx.format(i, site) for i in range(0, gate_num)] + [idx.format("f", site)]
	return l


def generate_pyramid_operators(link: int, n_op: int, left_idx, right_idx):
	out = qtn.TensorNetwork([])
	mod = link % 2
	if (
		n_op == 1 and len(right_idx) == 2
	):  # attempt to detect odd number of site boundary condition with just local information
		right_idx2 = right_idx
	else:
		right_idx2 = right_idx[mod::]
	for i in range(n_op):
		out &= qtn.Tensor(
			data=jnp.eye(4, 4).reshape(2, 2, 2, 2),
			inds=[
				left_idx[2 * i + mod],
				right_idx2[2 * i],
				left_idx[2 * i + mod + 1],
				right_idx2[2 * i + 1],
			],
			tags=["O", "O{}{}".format(link, n_op - i - 1)],
		)
	return out


def los_idx(n_op: int, site: int, nsite: int):
	idx = "l{}q{}"
	nlink = nsite - 1
	out = [idx.format(i, site) for i in range(n_op, -1, -1)]
	out[0] = idx.format("f", site)
	return out


def generate_losange_operators(link: int, midlink: int, n_op: int, left_idx, right_idx):
	# We make the losange downward pyramid first, and the pyramid shell first.
	if (len(left_idx) - len(right_idx)) >= 2:
		right_idx = [
			right_idx[0],
			*[x for r in right_idx[1:-1] for x in [r, r]],
			right_idx[-1],
		]
	elif -(len(left_idx) - len(right_idx)) >= 2:
		left_idx = [
			left_idx[0],
			*[x for r in left_idx[1:-1] for x in [r, r]],
			left_idx[-1],
		]
	out = qtn.TensorNetwork([])
	if link > midlink and len(left_idx) > 2:
		left_idx2 = left_idx[1:]
	else:
		left_idx2 = left_idx
	if link < midlink and len(right_idx) > 2:
		right_idx2 = right_idx[1:]
	else:
		right_idx2 = right_idx
	for i in range(n_op):
		out &= qtn.Tensor(
			data=jnp.eye(4, 4).reshape(2, 2, 2, 2),
			inds=[
				left_idx2[2 * i + 1],
				right_idx2[2 * i + 1],
				left_idx2[2 * i],
				right_idx2[2 * i],
			],
			tags=["O", "O{}{}".format(link, n_op - i - 1)],
		)
	return out


class staircaselayer:
	def __init__(self, data=np.random.rand(4, 4), uuid=qtn.rand_uuid() + "{}"):
		self.data = data.reshape(2, 2, 2, 2)
		self.bond_ind = uuid

	def __call__(
		self,
		input_idx,
		output_idx,
		Nlink,
		min_layer_number=0,
		dtype=jnp.float64,
		op_tag="Op{}",
		layer_tag="L",
		left_right_index=False,
	) -> Union[qtn.TensorNetwork, tuple[qtn.TensorNetwork, dict, dict]]:
		out = qtn.TensorNetwork([])
		left_inds = {}
		right_inds = {}
		i = 0
		if Nlink > 1:
			left = [input_idx.format(i), input_idx.format(i + 1)]
			right = [output_idx.format(i), self.bond_ind.format(i + 1)]
			utag = op_tag.format(i + min_layer_number)
			left_inds[utag] = left
			right_inds[utag] = right
			out &= qtn.Tensor(
				data=jnp.copy(self.data),
				inds=[*left, *right],
				tags=[layer_tag, utag, "O{},{}".format(i + min_layer_number, i)],
			)
			for i in range(1, Nlink - 1):
				left = [self.bond_ind.format(i), input_idx.format(i + 1)]
				right = [output_idx.format(i), self.bond_ind.format(i + 1)]
				utag = op_tag.format(i + min_layer_number)
				left_inds[utag] = left
				right_inds[utag] = right
				out &= qtn.Tensor(
					data=jnp.copy(self.data),
					inds=[*left, *right],
					tags=[layer_tag, utag, "O{},{}".format(i + min_layer_number, i)],
				)
			i = Nlink - 1
			left = self.bond_ind.format(i), input_idx.format(i + 1)
			right = output_idx.format(i), output_idx.format(i + 1)
			utag = op_tag.format(i + min_layer_number)
			left_inds[utag] = left
			right_inds[utag] = right
			out &= qtn.Tensor(
				data=jnp.copy(self.data),
				inds=[*left, *right],
				tags=[layer_tag, utag, "O{},{}".format(i + min_layer_number, i)],
			)
		else:
			left = [input_idx.format(i), input_idx.format(i + 1)]
			right = [output_idx.format(i), output_idx.format(i + 1)]
			utag = op_tag.format(i + min_layer_number)
			left_inds[utag] = left
			right_inds[utag] = right
			out &= qtn.Tensor(
				data=jnp.copy(self.data),
				inds=[*left, *right],
				tags=[layer_tag, utag, "O{},{}".format(i + min_layer_number, i)],
			)
		if left_right_index:
			return out, left_inds, right_inds
		else:
			return out

	@classmethod
	def toMPO(
		Class,
		layer: qtn.TensorNetwork,
		input_idx: str,
		output_idx: str,
		bond_idx: str,
		single_tensor_tag,
	):
		"""convert a staircase layer to a MPO."""
		Out_tens = []
		L = single_tensor_tag
		t = layer[L.format(0)]
		tl, tr = t.split(
			left_inds=[input_idx.format(0), output_idx.format(0)],
			absorb="both",
			cutoff=0.0,
		)
		inds = [tl.inds[-1], *tl.inds[:-1]]
		tl.transpose_(*inds)
		Out_tens.append(tl)
		Out_tens.append(tr)
		for f, _ in enumerate(layer.tensors[1:]):
			i = f + 1
			t = layer[L.format(i)]
			tl, tr = t.split(
				left_inds=[bond_idx.format(i), output_idx.format(i)],
				absorb="both",
				cutoff=0.0,
			)
			Out_tens[-1] = Out_tens[-1] @ tl
			inds = [
				Out_tens[-1].inds[0],
				Out_tens[-1].inds[-1],
				*Out_tens[-1].inds[1:-1],
			]
			Out_tens[-1].transpose_(*inds)
			Out_tens.append(tr)
		return qtn.MatrixProductOperator(
			[t.data for t in Out_tens],
			shape="lrdu",
			upper_ind_id=output_idx,
			lower_ind_id=input_idx,
		)  # À vérifier.


stl = staircaselayer()


def generate_staircase_operators(
	input_idx, output_idx, Nlink, min_layer_number, dtype=jnp.float64
):
	return stl(input_idx, output_idx, Nlink, min_layer_number, dtype=dtype)


def TwoqbitsStaircaseLayers(mps: qtn.TensorNetwork1D, Nlayer: int):
	L = len(mps.tensor_map)  # number of qbits
	out = qtn.TensorNetwork([])

	for layer in range(0, Nlayer - 1):
		out &= generate_staircase_operators(
			"l{}q{}".format(layer, "{}"), "l{}q{}".format(layer + 1, "{}"), L - 1, layer
		)
	out &= generate_staircase_operators(
		"l{}q{}".format(Nlayer - 1, "{}"), "l{}q{}".format("f", "{}"), L - 1, Nlayer - 1
	)
	return out


def TwoqbitsPyramidLayers(mps: qtn.TensorNetwork1D, Nlayer: int):
	L = len(mps.tensor_map) - 2  # number of qbits
	assert mps.ind_size("b1") <= 2
	left_idx = los_idx(Nlayer, 0, L)
	out = qtn.TensorNetwork([])
	for link in range(0, L - 1):
		ngate = Nlayer + Nlayer * (link != L - 2)
		right_idx = los_idx(ngate, link + 1, L)
		out &= generate_losange_operators(
			link, (L - 1) // 2, Nlayer, left_idx, right_idx
		)
		left_idx = right_idx
	return out


def TwoqbitsLosange(mps: qtn.TensorNetwork1D):
	L = len(mps.tensor_map) - 2  # number of qbits
	assert mps.ind_size("b1") <= 2
	back_bond = mps.ind_size("b{}".format(0))
	prev_ngate = int(np.ceil(np.log2(back_bond)))
	forward_bond = mps.ind_size("b{}".format(1))
	ngate = int(np.ceil(np.log2(forward_bond)))
	left_idx = los_idx(prev_ngate + ngate, 0, L)
	out = qtn.TensorNetwork([])
	for link in range(0, L - 1):
		back_bond = forward_bond
		forward_bond = mps.ind_size("b{}".format(link + 2))
		prev_ngate = ngate
		ngate = int(np.ceil(np.log2(forward_bond)))
		assert (
			jnp.abs(prev_ngate - ngate) <= 1
		)  # assert we're dealing with a properly compressed state
		right_idx = los_idx(prev_ngate + ngate, link + 1, L)
		out &= generate_losange_operators(
			link, (L - 1) // 2, prev_ngate, left_idx, right_idx
		)
		left_idx = right_idx
	return out


def generate_Lagrange_multipliers(Onet: qtn.TensorNetwork):
	out = qtn.TensorNetwork([])
	for t in Onet:
		tag = ""
		for at in t.tags:
			s = re.search("O(.+)", at)
			if s:
				tag = s.groups(0)[0]
		out &= qtn.Tensor(
			data=jnp.array(np.random.rand(2, 2, 2, 2)),
			inds=[
				"v{}".format(tag),
				"w{}".format(tag),
				"y{}".format(tag),
				"z{}".format(tag),
			],
			tags=["L", "L{}".format(tag)],
		)
	return out


def Infidelity(Onet, mps, triv_state):
	""" """
	S = triv_state | Onet
	fidelity = (S | mps).contract()  # optimize=opt)
	Nm = 1
	No = (S | S.H).contract()  # optimize=opt)
	return Nm + No - 2 * jnp.real(fidelity)


def unitarity_cost(
	Onet: qtn.TensorNetwork, Lag_mult: qtn.TensorNetwork, id: qtn.Tensor
):
	"""using this often means saddle point search instead of strict minimization"""
	cost = jnp.full([], 0)
	for (o, l) in zip(Onet, Lag_mult):
		oh_inds = dict(
			{ind: ind for ind in o.inds[0:2]},
			**{oind: lind for oind, lind in zip(o.inds[2:4], l.inds[2:4])}
		)
		o_inds = dict(
			{ind: ind for ind in o.inds[0:2]},
			**{oind: lind for oind, lind in zip(o.inds[2:4], l.inds[0:2])}
		)
		id.reindex_({idind: lind for idind, lind in zip(id.inds, l.inds)})
		oh = o.H.reindex_(oh_inds)
		o = o.reindex(o_inds)
		cost += l @ (id - oh @ o)
	return cost


def unitarity_cost2(
	Onet: qtn.TensorNetwork, Lag_mult: qtn.TensorNetwork, id: qtn.Tensor
):
	"""using this often means saddle point search instead of strict minimization"""
	cost = jnp.full([], 0)
	for (o, l) in zip(Onet, Lag_mult):
		oh_inds = dict(
			{ind: ind for ind in o.inds[0:2]},
			**{oind: lind for oind, lind in zip(o.inds[2:4], l.inds[2:4])}
		)
		o_inds = dict(
			{ind: ind for ind in o.inds[0:2]},
			**{oind: lind for oind, lind in zip(o.inds[2:4], l.inds[0:2])}
		)
		id.reindex_({idind: lind for idind, lind in zip(id.inds, l.inds)})
		oh = o.H.reindex_(oh_inds)
		o = o.reindex(o_inds)
		idmoo = id - oh @ o
		cost += idmoo.conj() @ idmoo
	return cost


def unitarity_test(
	Onet: qtn.TensorNetwork, Lag_mult: qtn.TensorNetwork, id: qtn.Tensor
):
	"""using this often means saddle point search instead of strict minimization"""
	cost = jnp.full([2, 2, 2, 2], 0)
	for (o, l) in zip(Onet, Lag_mult):
		oh_inds = dict(
			{ind: ind for ind in o.inds[0:2]},
			**{oind: lind for oind, lind in zip(o.inds[2:4], l.inds[2:4])}
		)
		o_inds = dict(
			{ind: ind for ind in o.inds[0:2]},
			**{oind: lind for oind, lind in zip(o.inds[2:4], l.inds[0:2])}
		)
		id.reindex_({idind: lind for idind, lind in zip(id.inds, l.inds)})
		oh = o.H.reindex_(oh_inds)
		o = o.reindex(o_inds)
		cost += (id - oh @ o).data
	return cost


def gauge_regularizer(Onet, id, C, m):
	cost = jnp.full([], 0)
	for o in Onet:
		id.reindex_({idind: lind for idind, lind in zip(id.inds, o.inds)})
		x = id - o
		xx = x.conj() @ x
		cost += C * jnp.log(1 + m * xx)
	return cost


def magic_loss(tn, psi, trivial_state, id, C, m):
	"""combine the fidelity loss, lagrange multiplier, and gauge reularizer. the set of 2 qbits gate and lagrange multiplier must be supplied as a single network with tags 'O' and 'L' to distinguish between the two"""
	Onet = qtn.TensorNetwork(tn["O"])
	Lag_mult = qtn.TensorNetwork(tn["L"])
	return (
		Infidelity(Onet, psi, trivial_state) + unitarity_cost(Onet, Lag_mult, id)
	) * (1 + gauge_regularizer(Onet, id, C, m))


def positive_magic_loss(tn, psi, trivial_state, id, C, m):
	"""combine the fidelity loss, lagrange multiplier, and gauge reularizer. the set of 2 qbits gate and lagrange multiplier must be supplied as a single network with tags 'O' and 'L' to distinguish between the two"""
	Onet = qtn.TensorNetwork(tn["O"])
	Lag_mult = qtn.TensorNetwork(tn["L"])
	uc = unitarity_cost(Onet, Lag_mult, id)
	return (Infidelity(Onet, psi, trivial_state) + jnp.abs(uc)) * (
		1 + gauge_regularizer(Onet, id, C, m)
	)


def magic_loss2(tn, psi, trivial_state, L, id, C, m):
	"""No lagrange multiplier for this one, unitarity constraint must be imposed by some other means."""
	return (Infidelity(tn, psi, trivial_state) + unitarity_cost2(tn, L, id)) * (
		1 + gauge_regularizer(tn, id, C, m)
	)


def trivial_state(nqbit: int, label="l0q{}", bin=0):
	return qtn.TensorNetwork(
		[
			qtn.Tensor(
				data=jnp.array(
					[
						1 * ((1 << (nqbit - 1 - i)) & bin == 0),
						1 * ((1 << (nqbit - 1 - i)) & bin != 0),
					]
				),
				tags="TS",
				inds=[label.format(i)],
			)
			for i in range(nqbit)
		]
	)


def randomize_net(net):
	out = qtn.TensorNetwork([])
	for t in net:
		out &= qtn.Tensor(
			data=jnp.array(np.random.rand(*t.data.shape)), inds=t.inds, tags=t.tags
		)
	return out


def normalize_gate(gate: qtn.Tensor):
	U, d, V = qtn.tensor_core.tensor_split(
		gate, gate.inds[0:2], absorb=None, get="tensors", cutoff=0.0
	)
	return U @ V


def normalize_gates(gate_set: qtn.TensorNetwork):
	out = qtn.TensorNetwork([])
	for gate in gate_set:
		out &= normalize_gate(gate)
	return out


def MPS2gates2(
	mps: qtn.MatrixProductState, precision: float, max_layer: int, dtype=np.float64
):
	"""
	Will proceed in the following manner:
	1. optimize a single layer of staircase gates against the current MPS
	    1.1 accumulate the layer in the output structure
	2. convert staircase in MPO
	3. update the MPS by contracting hermitian conjugate of MPO with MPS (and compress)
	4. Go to 1. if error is greater than target precision and max_layer is not reached
	5. return the output structure to the caller
	"""
	out = qtn.TensorNetwork([])
	mps = mps.copy(deep=True)
	layer_generator = staircaselayer(np.eye(4))
	in_idx_L = qtn.rand_uuid() + "{}"
	for i in range(max_layer):
		in_idx = in_idx_L.format(i) + "{}"
		Zero_state = trivial_state(mps.L, in_idx)
		utag = "Op{}"
		Ltag = "L{}".format(i)
		layer, left_inds, right_inds = layer_generator(
			in_idx,
			mps.site_ind_id,
			mps.L - 1,
			0,
			dtype,
			utag,
			Ltag,
			left_right_index=True,
		)
		# for the SVD optimizer to stop as early as possible, il faut d'abord réduire la dimension de lien du MPS à deux..
		# p-e qu'il est possible d'obtenir de meilleur résultat en limitant simplement le nombre de sweep. so far,so bad
		# 1.
		cmps = mps.copy()
		layer, optimizer_err = layer_SVD_optimizer(
			layer,
			left_inds,
			Zero_state,
			cmps,
			utag,
			max_it=20,
			prec=1e-13,
			renorm=False,
			return_error=True,
		)
		out &= layer
		# 2.
		layer_mpo = staircaselayer.toMPO(
			layer, in_idx, mps.site_ind_id, layer_generator.bond_ind, utag
		)
		layer_mpo = layer_mpo.partial_transpose([*range(layer_mpo.L)]).H
		# 3.
		layer_mpo.site_tag_id = mps.site_tag_id
		new_mps = MPS_compression((layer_mpo.H).apply(mps), 1, 1e-13, 1e-13)
		# 4.
		err = np.sqrt(1 - np.abs((mps | layer.H | Zero_state).contract()))
		print("error: ", err)
		if err < precision:
			break
		mps = new_mps
		mps.site_ind_id = in_idx

	return out, err


def MPS2Gates(mps, precision, Nlayer, max_count=40):
	qX = mps
	O = TwoqbitsStaircaseLayers(qX, Nlayer)
	L = generate_Lagrange_multipliers(O)
	ts = trivial_state(mps.L)
	id = qtn.Tensor(data=jnp.eye(4, 4).reshape(2, 2, 2, 2), inds=("a", "b", "c", "d"))
	optmzr = TNOptimizer(
		O,
		loss_fn=magic_loss2,
		# norm_fn=normalize_gates,
		loss_kwargs={"C": 0.1, "m": 50},
		loss_constants={
			"psi": qX,
			"trivial_state": ts,
			"id": id,
			"L": L,
		},  # this is a constant TN to supply to loss_fn: psi,trivial_state, id, C,m)
		autodiff_backend="jax",  # {'jax', 'tensorflow', 'autograd','torch'}
		optimizer="L-BFGS-B",
		loss_target=precision,
	)
	error = 1000
	count = 0
	OL_opt = optmzr.optimize(100)
	while error > precision and count < max_count:
		optmzr.reset(OL_opt, loss_target=precision)
		val, grad = optmzr.vectorized_value_and_grad(optmzr.vectorizer.vector)
		print("initial gradient: ", np.mean(np.abs(grad)))
		OL_opt = optmzr.optimize(20000, tol=precision)
		error = Infidelity(OL_opt, qX, ts)
		print(
			"current error: ",
			error,
			" unitarity error: ",
			unitarity_cost2(OL_opt, L, id),
		)
		count += 1
	O = normalize_gates(OL_opt)
	return O, error


def optimize_unitaries(
	Onet: qtn.TensorNetwork,
	tags_re,
	stateA: qtn.TensorNetwork,
	stateB: qtn.TensorNetwork,
	precision,
	max_iteration=1000,
):
	"""Gradient free, very slow convergence..."""

	def filt(x):
		return re.search(tags_re, x)

	Opt_tags = [*filter(filt, Onet.tags)]
	stateA.add_tag("STATEA")
	stateB.add_tag("STATEB")
	Scalar_Net = Onet | stateA | stateB
	# Scalar_Net.draw()
	print("starting fidelity: ", Scalar_Net.contract())
	fid = 10000
	new_fid = 10000
	count = 0
	print("====OPTIMIZE UNITARIES====")
	while (abs(1 - fid) > precision) and (count < max_iteration):
		fid = new_fid
		for tag in Opt_tags:
			Scalar_Net = Onet | stateA | stateB
			Otag = Scalar_Net[tag].tags
			tags = Scalar_Net.tags.difference(qtn.oset(Otag))
			V = Scalar_Net.contract(tags)  # ,optimize=opt)
			R = V[tags]
			u, d, v = qtn.tensor_split(
				R, R.inds[:2], get="tensors", absorb=None, cutoff=0.0
			)
			tmp = v.H @ u.H
			tmp.retag_({t: None for t in tmp.tags})
			for Ot in Otag:
				tmp.tags.add(Ot)
			Onet[tag] = tmp
			new_fid = d.sum_reduce(d.inds[0]).data
			print("tag {} iteration {}, fidelity {}".format(tag, count, new_fid))
		count += 1
	return Onet
