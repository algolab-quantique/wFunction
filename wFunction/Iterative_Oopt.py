try:
	import cotengra as ctg

	opt = ctg.ReusableHyperOptimizer(
		reconf_opts={},
		max_repeats=16,
		parallel=True,
		# use the following for persistently cached paths
		# directory='ctg_path_cache',
	)
	nocontengra = False
except:
	nocontengra = True

import quimb.tensor as qtn
import re


def optimize_unitaries(Onet, tags_re, stateA, stateB, precision, max_iteration=1000):
	def filt(x):
		return re.search(tags_re, x)

	Opt_tags = filter(filt, Onet.tags)
	Scalar_Net = Onet | stateA | stateB
	fid = 10000
	count = 0
	while (1 - fid) > precision and count < max_iteration:
		for tag in Opt_tags:
			tags = Scalar_Net.tags.difference(qtn.oset(tag))
			if nocontengra:
				V = Scalar_Net.contract(tags)
			else:
				V = Scalar_Net.contract(tags, optimize=opt)
			R = V[tags]
			u, d, v = qtn.split_tensor(R, R.inds[:2], get="tensors", absorb=None)
			V[tag] = u @ v

		fid = d.sum_reduce(d.inds[0]).data
	return Onet
