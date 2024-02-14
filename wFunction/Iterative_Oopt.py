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
