
from jax.config import config
from torch import zero_
config.update("jax_enable_x64", True)

import quimb as qb
import quimb.tensor as qtn
import quantit as qtt
import numpy as np
import jax.numpy as jnp
from quimb.tensor.optimize import TNOptimizer
import re
# import cotengra as ctg

# opt = ctg.ReusableHyperOptimizer(
#     reconf_opts={},
#     max_repeats=16,
#     parallel=True,
#     # use the following for persistently cached paths
#     # directory='ctg_path_cache',
# )


from compress_algs import print_sizes


TN = qtn.TensorNetwork
def qttMPS2quimbTN(mps:qtt.networks.MPS, phys_label:str):
    b = 'b{}'
    cap0 = qtn.Tensor(data = jnp.ones([mps[0].size()[0]]),inds=(b.format(0),))
    capN = qtn.Tensor(data = jnp.ones([mps[0].size()[0]]),inds=(b.format(len(mps)),))
    return cap0 & qtn.TensorNetwork1D( [ qtn.Tensor(data = jnp.array(t.numpy()), inds = (b.format(i), phys_label+str(i),b.format(i+1)), tags=['t'+str(i)] ) for i,t in enumerate(mps)]) & capN 

def log4(x):
    return jnp.log(x)/jnp.log(4)

def link_pyramid_gate_num(bond_size:int,bond:int):
    if bond_size <= 1:
        return 0
    if bond%2==0:
        return int(jnp.ceil((log4(bond_size/2) if bond_size>2 else 0)+1))
    else:
        return int(jnp.ceil(log4(bond_size)))

def idx_list(site:int,gate_num:int):
    idx = 'l{}q{}'
    l = [idx.format(i,site) for i in range(0,gate_num)] + [idx.format('f',site)]
    return l

def generate_operators(link:int,bond_size:int,left_idx,right_idx):
    n_op = link_pyramid_gate_num(bond_size,link)
    out = qtn.TensorNetwork([])
    mod = (link%2)
    if n_op == 1 and len(right_idx)==2: # attempt to detect odd number of site boundary condition with just local information
        right_idx2 = right_idx
    else:
        right_idx2 = right_idx[mod::]
    for i in range(n_op):
        out &= qtn.Tensor(data =jnp.eye(4,4).reshape(2,2,2,2), inds = [left_idx[2*i+mod],right_idx2[2*i], left_idx[2*i+mod+1],right_idx2[2*i+1]],tags=['O','O{}{}'.format(link,n_op-i-1)])
    return out


def TwoqbitsPyramid(mps:qtn.TensorNetwork1D):
    #this implementation doesn't work, for a few reason: 
    # - we should be taking the log4 of the bond dim instead of log2 to determine the number of 2qbits gate necessary for a given link.
    # - The network must be built by the base of the pyramid then the tip, and left hollow if no more gates are necessary. The tip must contract with the MPS first.
    # - we must take care to label the index correctly when an odd number of qbits are used.
    # - if a zero gate situation happen, it mean we should put a 1qbit gate on the site the closest to an edge.
    L = len(mps.tensor_map)-2 #number of qbits
    assert(mps.ind_size('b1') <= 2)
    back_bond = mps.ind_size('b{}'.format(0))
    prev_ngate = link_pyramid_gate_num(back_bond)
    forward_bond = mps.ind_size('b{}'.format(1))
    left_idx = idx_list(0,back_bond,forward_bond)
    out = qtn.TensorNetwork([])
    for link in range(0,L - 1 ):
        ngate = link_pyramid_gate_num(forward_bond,link)
        assert(jnp.abs(prev_ngate - ngate)<=1)#assert we're dealing with a properly compressed state
        back_bond = forward_bond
        forward_bond = mps.ind_size('b{}'.format(link+2))
        right_idx = idx_list(link+1,back_bond,forward_bond)
        out &= generate_operators(link,back_bond,left_idx,right_idx)
        back_bond = forward_bond
        left_idx = right_idx
        prev_ngate = ngate
    return out

def twobitsLosange(mps:qtn.TensorNetwork1D):
    L = len(mps.tensor_map)-2 #number of qbits
    assert(mps.ind_size('b1') <= 2)
    back_bond = mps.ind_size('b{}'.format(0))
    prev_ngate = jnp.log2(back_bond,-1)
    forward_bond = mps.ind_size('b{}'.format(1))
    left_idx = idx_list(0,back_bond,forward_bond)
    out = qtn.TensorNetwork([])

def generate_Lagrange_multipliers(Onet:qtn.TensorNetwork):
    out = qtn.TensorNetwork([])
    for t in Onet:
        tag = ''
        for at in t.tags:
            s = re.search('O(.+)',at)
            if s:
                tag = s.groups(0)[0]
        out &= qtn.Tensor(data = jnp.array(np.random.rand(2,2,2,2)),inds = ['v{}'.format(tag),'w{}'.format(tag),'y{}'.format(tag),'z{}'.format(tag)], tags=['L','L{}'.format(tag)])
    return out

def Infidelity(Onet,mps,triv_state):
    """
    negative log of the fidelity. zero when the objects match, infinite when fidelity is zero.
    """
    S = triv_state|Onet
    fidelity = (S|mps).contract()#optimize=opt)
    Nm = 1
    No = (S|S.H).contract()#optimize=opt)
    # print(Onet['O00'].data.dtype)
    return (Nm+No - 2*jnp.real(fidelity))


def unitarity_cost(Onet:qtn.TensorNetwork,Lag_mult:qtn.TensorNetwork,id:qtn.Tensor):
    """ using this often means saddle point search instead of strict minimization"""
    cost = jnp.full([],0)
    for (o,l) in zip(Onet,Lag_mult):
        oh_inds = dict({ind:ind for ind in o.inds[0:2]}, **{ oind:lind for oind,lind in zip(o.inds[2:4],l.inds[2:4])})
        o_inds = dict({ind:ind for ind in o.inds[0:2]}, **{  oind:lind for oind,lind in zip(o.inds[2:4],l.inds[0:2])})
        id.reindex_({idind:lind for idind,lind in zip(id.inds,l.inds)})
        oh = o.H.reindex_(oh_inds)
        o = o.reindex(o_inds)
        cost += l@(id-oh@o)
    return cost

def unitarity_cost2(Onet:qtn.TensorNetwork,Lag_mult:qtn.TensorNetwork,id:qtn.Tensor):
    """ using this often means saddle point search instead of strict minimization"""
    cost = jnp.full([],0)
    for (o,l) in zip(Onet,Lag_mult):
        oh_inds = dict({ind:ind for ind in o.inds[0:2]}, **{ oind:lind for oind,lind in zip(o.inds[2:4],l.inds[2:4])})
        o_inds = dict({ind:ind for ind in o.inds[0:2]}, **{  oind:lind for oind,lind in zip(o.inds[2:4],l.inds[0:2])})
        id.reindex_({idind:lind for idind,lind in zip(id.inds,l.inds)})
        oh = o.H.reindex_(oh_inds)
        o = o.reindex(o_inds)
        idmoo = id-oh@o
        cost += idmoo.conj()@idmoo
    return cost

def unitarity_test(Onet:qtn.TensorNetwork,Lag_mult:qtn.TensorNetwork,id:qtn.Tensor):
    """ using this often means saddle point search instead of strict minimization"""
    cost = jnp.full([2,2,2,2],0)
    for (o,l) in zip(Onet,Lag_mult):
        oh_inds = dict({ind:ind for ind in o.inds[0:2]}, **{ oind:lind for oind,lind in zip(o.inds[2:4],l.inds[2:4])})
        o_inds = dict({ind:ind for ind in o.inds[0:2]}, **{  oind:lind for oind,lind in zip(o.inds[2:4],l.inds[0:2])})
        id.reindex_({idind:lind for idind,lind in zip(id.inds,l.inds)})
        oh = o.H.reindex_(oh_inds)
        o = o.reindex(o_inds)
        cost += (id-oh@o).data
    return cost


def gauge_regularizer(Onet,id, C,m):
    cost = jnp.full([],0)
    for o in Onet:
        id.reindex_({idind:lind for idind,lind in zip(id.inds,o.inds)})
        x = id - o
        xx = x.conj()@x
        cost += C*jnp.log(1+m*xx)
    return cost

def magic_loss(tn, psi,trivial_state, id, C,m):
    """combine the fidelity loss, lagrange multiplier, and gauge reularizer. the set of 2 qbits gate and lagrange multiplier must be supplied as a single network with tags 'O' and 'L' to distinguish between the two"""
    Onet = qtn.TensorNetwork(tn['O'])
    Lag_mult = qtn.TensorNetwork(tn['L'])
    return Infidelity(Onet,psi,trivial_state) + unitarity_cost(Onet,Lag_mult,id) #+ gauge_regularizer(Onet,id,C,m)

def positive_magic_loss(tn, psi,trivial_state, id, C,m):
    """combine the fidelity loss, lagrange multiplier, and gauge reularizer. the set of 2 qbits gate and lagrange multiplier must be supplied as a single network with tags 'O' and 'L' to distinguish between the two"""
    Onet = qtn.TensorNetwork(tn['O'])
    Lag_mult = qtn.TensorNetwork(tn['L'])
    uc = unitarity_cost(Onet,Lag_mult,id)
    return Infidelity(Onet,psi,trivial_state) + jnp.abs(uc) #+ gauge_regularizer(Onet,id,C,m)

def magic_loss2(tn,psi,trivial_state,L,id,C,m):
    """No lagrange multiplier for this one, unitarity constraint must be imposed by some other means."""
    return Infidelity(tn,psi,trivial_state) + unitarity_cost2(tn,L,id)# + gauge_regularizer(tn,id,C,m)

def trivial_state(nqbit:int,label='l0q',bin=0):
    return qtn.TensorNetwork([qtn.Tensor(data = jnp.array([1*((1<<(nqbit-1-i))&bin == 0),1*((1<<(nqbit-1-i))&bin != 0)]), inds=[label+'{}'.format(i)]) for i in range(nqbit)])

def randomize_net(net):
    out = qtn.TensorNetwork([])
    for t in net:
        out &=qtn.Tensor(data = jnp.array(np.random.rand(*t.data.shape)),inds= t.inds,tags = t.tags)
    return out

def normalize_gate(gate:qtn.Tensor):
    U,d,V = qtn.tensor_core.tensor_split(gate,gate.inds[0:2],absorb=None,get='tensors',cutoff=0.0)
    return U@V

def normalize_gates(gate_set:qtn.TensorNetwork):
    out = qtn.TensorNetwork([])
    for gate in gate_set:
        out &= normalize_gate(gate)
    return out

def MPS2Gates(mps,precision,max_count=1000):
    qX = qttMPS2quimbTN(mps,'lfq')
    O = TwoqbitsPyramid(qX)
    L = generate_Lagrange_multipliers(O)
    ts = trivial_state(len(mps))
    id = qtn.Tensor(data = jnp.eye(4,4).reshape(2,2,2,2), inds=('a','b','c','d'))
    optmzr = TNOptimizer(
        O,
        loss_fn = magic_loss2,
        # norm_fn=normalize_gates,
        loss_kwargs = {'C':0,'m':50},
        loss_constants={'psi': qX, 'trivial_state':ts,'id':id,'L':L},  # this is a constant TN to supply to loss_fn: psi,trivial_state, id, C,m)
        autodiff_backend='jax',      # {'jax', 'tensorflow', 'autograd','torch'}
        optimizer='L-BFGS-B',
        loss_target=precision
    )
    error = 1000
    count = 0
    print(gauge_regularizer(O,id,1/50,50))
    print(Infidelity(O,qX,ts))
    OL_opt = optmzr.optimize(100)    
    # OL_opt = normalize_gates(TN(OL_opt['O']))&TN(OL_opt['L'])
    while error>precision and count <max_count:
        optmzr.reset(OL_opt,loss_target=precision)
        val,grad = optmzr.vectorized_value_and_grad(optmzr.vectorizer.vector)
        print("initial gradient: ",np.mean(np.abs(grad)) )
        # if (count%3==1):
        #     OL_opt = optmzr.optimize_basinhopping(n=200, nhop=10,temperature=0.5)
        # else:
        OL_opt = optmzr.optimize(20000,tol=precision)   
        # OL_opt = normalize_gates(TN(OL_opt['O']))#&normalize_gates(TN(OL_opt['L']))
        error = Infidelity(TN(OL_opt['O']),qX,ts)
        # print("count: ", count)
        print("current error: ", error, " unitarity error: ", unitarity_cost2(OL_opt,L,id))
        count += 1 
    O = normalize_gates(OL_opt)
    return O,error

def optimize_unitaries(Onet:qtn.TensorNetwork,tags_re,stateA:qtn.TensorNetwork,stateB:qtn.TensorNetwork,precision,max_iteration=1000):
    def filt(x):
        return re.search(tags_re,x)
    Opt_tags = filter(filt,Onet.tags)
    stateA.add_tag("STATEA")
    stateB.add_tag("STATEB")
    Scalar_Net = Onet|stateA|stateB
    print("starting fidelity: ", Scalar_Net.contract()) 
    fid = 10000
    new_fid=0
    count = 0
    print("====OPTIMIZE UNITARIES====")
    while abs(new_fid-fid)> precision and count < max_iteration: 
        fid = new_fid
        for tag in Opt_tags:
            Otag = Scalar_Net[tag].tags
            tags = Scalar_Net.tags.difference(qtn.oset(Otag))
            print("tags: ", tags)
            print("tag: ", tag)
            V = Scalar_Net.contract(tags)#,optimize=opt)
            print("V:", V)
            R = V[tags]
            print(R.inds)
            u,d,v = qtn.tensor_split(R,R.inds[:2],get='tensors',absorb=None)
            V[tag] = v.H@u.H
            new_fid = d.sum_reduce(d.inds[0]).data
            print("tag {} iteration {}, fidelity {}".format(tag,count,new_fid))
        count +=1
    return Onet

if __name__=='__main__':
    import compress_algs as calgs
    import interpolate as terp
    import Generate_circuit as gerc
    import matplotlib.pyplot as plt
    import seaborn as sb
    sb.set_theme()

    def f(x):
        return np.exp(-x**2)
    nqbit = 10
    domain = (-1,1)
    Gate_precision = 1e-12
    MPS_precision = 0.001
    polys = gerc.poly_by_part(f,0.001,nqbit,domain)
    mpses = [terp.polynomial2MPS(poly,nqbit,pdomain,domain) for poly,pdomain in polys]
    X = calgs.MPS_compressing_sum(mpses,0.1*MPS_precision,MPS_precision)
    print_sizes(X)
    for t in X:
        print(t.size())
    X[X.orthogonality_center] /= np.sqrt(qtt.networks.contract(X,X))
    O, inf = MPS2Gates(X,Gate_precision,max_count=0)
    zero_state = trivial_state(nqbit)
    qX = qttMPS2quimbTN(X,'lfq')
    O = optimize_unitaries(O,"O\d+",zero_state,qX,Gate_precision,max_iteration=100)
    O.draw()
    for t in qX:
        print(t)
    SO = []
    SM = []
    for t in O:
        print(t)
    print(zero_state)
    for binstate in range(2**nqbit):
        ts = trivial_state(nqbit,'lfq',binstate)
        Val = (zero_state&O)@ts
        SO.append(Val)
        SM.append(qX@ts)
    # print(SO)
    # print(SM)
    SO = np.array(SO)
    SM = np.array(SM)
    print((zero_state&O)@(zero_state&O))
    print(qX@qX)
    F = (zero_state&O)@qX
    print(F)
    print(np.dot(SO,SM))
    print((1-F)**2)
    plt.plot(SO)
    plt.plot(SM)
    plt.show()



    