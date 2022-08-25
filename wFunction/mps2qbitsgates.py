
from jax.config import config
config.update("jax_enable_x64", True)

import quimb as qb
import quimb.tensor as qtn
# import quantit as qtt
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


# from .compress_algs import print_sizes

qtn.MatrixProductState
TN = qtn.TensorNetwork
# def qttMPS2quimbTN(mps:qtt.networks.MPS, phys_label:str):
#     b = 'b{}'
#     cap0 = qtn.Tensor(data = jnp.ones([mps[0].size()[0]]),inds=(b.format(0),))
#     capN = qtn.Tensor(data = jnp.ones([mps[0].size()[0]]),inds=(b.format(len(mps)),))
#     return cap0 & qtn.TensorNetwork1D( [ qtn.Tensor(data = jnp.array(t.numpy()), inds = (b.format(i), phys_label+str(i),b.format(i+1)), tags=['t'+str(i)] ) for i,t in enumerate(mps)]) & capN 

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

def generate_pyramid_operators(link:int,n_op:int,left_idx,right_idx):
    # n_op = link_pyramid_gate_num(bond_size,link)
    out = qtn.TensorNetwork([])
    mod = (link%2)
    if n_op == 1 and len(right_idx)==2: # attempt to detect odd number of site boundary condition with just local information
        right_idx2 = right_idx
    else:
        right_idx2 = right_idx[mod::]
    for i in range(n_op):
        out &= qtn.Tensor(data =jnp.eye(4,4).reshape(2,2,2,2), inds = [left_idx[2*i+mod],right_idx2[2*i], left_idx[2*i+mod+1],right_idx2[2*i+1]],tags=['O','O{}{}'.format(link,n_op-i-1)])
    return out

def los_idx(n_op:int,site:int,nsite:int):
    idx = 'l{}q{}'
    nlink = nsite - 1
    midlink = nlink/2
    # maxi = int(nlink//2) + 1 - int(np.floor(abs( midlink - site ) ))
    out = [idx.format(i,site) for i in range(n_op,-1,-1)]
    out[0] = idx.format('f',site)
    return out


def generate_losange_operators(link:int,midlink:int,n_op:int,left_idx,right_idx):
    # We make the losange downward pyramid first, and the pyramid shell first.
    if (len(left_idx) - len(right_idx))>=2:
        right_idx = [right_idx[0] , *[  x  for r in right_idx[1:-1] for x in [r,r] ] ,right_idx[-1]]
    elif -(len(left_idx) - len(right_idx)) >=2:
        left_idx = [left_idx[0] , *[  x  for r in left_idx[1:-1] for x in [r,r] ] ,left_idx[-1]]
    out = qtn.TensorNetwork([])
    if link > midlink and len(left_idx) >2:
        left_idx2 = left_idx[1:]
    else:
        left_idx2 = left_idx
    if link < midlink and len(right_idx) > 2:
        right_idx2 = right_idx[1:]
    else:
        right_idx2 = right_idx
    for i in range(n_op):
        out &= qtn.Tensor(data =jnp.eye(4,4).reshape(2,2,2,2), inds = [left_idx2[2*i+1],right_idx2[2*i+1], left_idx2[2*i],right_idx2[2*i]],tags=['O','O{}{}'.format(link,n_op-i-1)])
    return out

class staircaselayer():
    def __init__(self,data = np.random.rand(4,4),uuid = qtn.rand_uuid() + '{}'):
        self.data = data.reshape(2,2,2,2)
        self.data /= np.sum(self.data**2)
        self.uuid = uuid
    def __call__(self,input_idx, output_idx,Nlink, min_layer_number=0,dtype=jnp.float64) -> qtn.TensorNetwork:
        out = qtn.TensorNetwork([])
        i = 0
        if Nlink > 1:
            out &= qtn.Tensor(data =jnp.copy(self.data), inds = [input_idx.format(i),input_idx.format(i+1),output_idx.format(i),self.uuid.format(i+1)],tags=['O','L{}'.format(i+min_layer_number),"O{},{}".format(i+min_layer_number,i)])
            for i in range(1,Nlink-1):
                out &= qtn.Tensor(data =jnp.copy(self.data), inds = [self.uuid.format(i),input_idx.format(i+1),output_idx.format(i),self.uuid.format(i+1)],tags=['O','L{}'.format(i+min_layer_number),"O{},{}".format(i+min_layer_number,i)])
            i = Nlink-1
            out &= qtn.Tensor(data =jnp.copy(self.data), inds = [self.uuid.format(i),input_idx.format(i+1),output_idx.format(i),output_idx.format(i+1)],tags=['O','L{}'.format(i+min_layer_number),"O{},{}".format(i+min_layer_number,i)])
        else:
            out &= qtn.Tensor(data =jnp.copy(self.data), inds = [input_idx.format(i),input_idx.format(i+1),output_idx.format(i),output_idx.format(i+1)],tags=['O','L{}'.format(i+min_layer_number),"O{},{}".format(i+min_layer_number,i)])
        return out

stl = staircaselayer()

def generate_staircase_operators(input_idx, output_idx,Nlink, min_layer_number,dtype=jnp.float64):
    return stl(input_idx, output_idx,Nlink, min_layer_number,dtype=dtype)

def TwoqbitsStaircaseLayers(mps:qtn.TensorNetwork1D,Nlayer:int):
    L = len(mps.tensor_map) #number of qbits
    out = qtn.TensorNetwork([])

    for layer in range(0,Nlayer-1 ):
        out &= generate_staircase_operators("l{}q{}".format(layer,"{}"),"l{}q{}".format(layer+1,"{}"),L-1,layer)
    out &= generate_staircase_operators("l{}q{}".format(Nlayer-1,"{}"),"l{}q{}".format("f","{}"),L-1,Nlayer-1)
    return out

def TwoqbitsPyramidLayers(mps:qtn.TensorNetwork1D,Nlayer:int):
    L = len(mps.tensor_map)-2 #number of qbits
    assert(mps.ind_size('b1') <= 2)
    left_idx = los_idx(Nlayer,0,L)
    out = qtn.TensorNetwork([])
    for link in range(0,L - 1 ):
        ngate = Nlayer + Nlayer*(link!=L-2)
        right_idx = los_idx(ngate,link+1,L)
        out &= generate_losange_operators(link,(L-1)//2,Nlayer,left_idx,right_idx)
        left_idx = right_idx
    return out

def TwoqbitsLosange(mps:qtn.TensorNetwork1D):
    L = len(mps.tensor_map)-2 #number of qbits
    assert(mps.ind_size('b1') <= 2)
    back_bond = mps.ind_size('b{}'.format(0))
    prev_ngate = int(np.ceil(np.log2(back_bond)))
    forward_bond = mps.ind_size('b{}'.format(1))
    ngate = int(np.ceil(np.log2(forward_bond)))
    left_idx = los_idx(prev_ngate+ngate,0,L)
    out = qtn.TensorNetwork([])
    for link in range(0,L - 1 ):
        back_bond = forward_bond
        forward_bond = mps.ind_size('b{}'.format(link+2))
        prev_ngate = ngate
        ngate = int(np.ceil(np.log2(forward_bond)))
        assert(jnp.abs(prev_ngate - ngate)<=1)#assert we're dealing with a properly compressed state
        right_idx = los_idx(prev_ngate+ngate,link+1,L)
        out &= generate_losange_operators(link,(L-1)//2,prev_ngate,left_idx,right_idx)
        left_idx = right_idx
    return out

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
    return (Infidelity(Onet,psi,trivial_state) + unitarity_cost(Onet,Lag_mult,id))*( 1 + gauge_regularizer(Onet,id,C,m))

def positive_magic_loss(tn, psi,trivial_state, id, C,m):
    """combine the fidelity loss, lagrange multiplier, and gauge reularizer. the set of 2 qbits gate and lagrange multiplier must be supplied as a single network with tags 'O' and 'L' to distinguish between the two"""
    Onet = qtn.TensorNetwork(tn['O'])
    Lag_mult = qtn.TensorNetwork(tn['L'])
    uc = unitarity_cost(Onet,Lag_mult,id) 
    return (Infidelity(Onet,psi,trivial_state) + jnp.abs(uc) )*(1 + gauge_regularizer(Onet,id,C,m))

def magic_loss2(tn,psi,trivial_state,L,id,C,m):
    """No lagrange multiplier for this one, unitarity constraint must be imposed by some other means."""
    return (Infidelity(tn,psi,trivial_state) + unitarity_cost2(tn,L,id))*(1 + gauge_regularizer(tn,id,C,m))

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

def MPS2Gates(mps,precision,Nlayer,max_count=40):
    qX = mps
    O = TwoqbitsStaircaseLayers(qX,Nlayer)
    L = generate_Lagrange_multipliers(O)
    ts = trivial_state(mps.L)
    # print("O",O)
    # print("X",qX)
    # print("ts",ts)
    id = qtn.Tensor(data = jnp.eye(4,4).reshape(2,2,2,2), inds=('a','b','c','d'))
    optmzr = TNOptimizer(
        O,
        loss_fn = magic_loss2,
        # norm_fn=normalize_gates,
        loss_kwargs = {'C':0.1,'m':50},
        loss_constants={'psi': qX, 'trivial_state':ts,'id':id,'L':L},  # this is a constant TN to supply to loss_fn: psi,trivial_state, id, C,m)
        autodiff_backend='jax',      # {'jax', 'tensorflow', 'autograd','torch'}
        optimizer='L-BFGS-B',
        loss_target=precision
    )
    error = 1000
    count = 0
    # print(gauge_regularizer(O,id,1/50,50))
    # print(Infidelity(O,qX,ts))
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
        error = Infidelity(OL_opt,qX,ts)
        # print("count: ", count)
        print("current error: ", error, " unitarity error: ", unitarity_cost2(OL_opt,L,id))
        count += 1 
    O = normalize_gates(OL_opt)
    return O,error

def optimize_unitaries(Onet:qtn.TensorNetwork,tags_re,stateA:qtn.TensorNetwork,stateB:qtn.TensorNetwork,precision,max_iteration=1000):
    """Gradient free, very slow convergence..."""
    def filt(x):
        return re.search(tags_re,x)
    Opt_tags = [*filter(filt,Onet.tags)]
    stateA.add_tag("STATEA")
    stateB.add_tag("STATEB")
    Scalar_Net = Onet|stateA|stateB
    # Scalar_Net.draw()
    print("starting fidelity: ", Scalar_Net.contract()) 
    fid = 10000
    new_fid=10000
    count = 0
    print("====OPTIMIZE UNITARIES====")
    while (abs(1-fid) > precision) and (count < max_iteration): 
        fid = new_fid
        for tag in Opt_tags:
            Scalar_Net = Onet|stateA|stateB
            Otag = Scalar_Net[tag].tags
            tags = Scalar_Net.tags.difference(qtn.oset(Otag))
            # print("tags: ", tags)
            # print("tag: ", tag)
            V = Scalar_Net.contract(tags)#,optimize=opt)
            # print("V:", V)
            R = V[tags]
            # print(R.inds)
            u,d,v = qtn.tensor_split(R,R.inds[:2],get='tensors',absorb=None,cutoff=0.0)
            tmp = (v.H@u.H)
            tmp.retag_({t:None for t in tmp.tags})
            # print(Onet[tag].data)
            for Ot in Otag:
                tmp.tags.add(Ot)
            Onet[tag] = tmp
            # print(Onet[tag].data)
            new_fid = d.sum_reduce(d.inds[0]).data
            print("tag {} iteration {}, fidelity {}".format(tag,count,new_fid))
        count +=1
    return Onet

if __name__=='__main__':
    #fiddling to make this file work when loaded directly...
    import sys
    import os 
    from pathlib import Path

    file = Path(__file__).resolve()
    parent, top = file.parent, file.parents[1]
    sys.path.append(str(top))
    try:
        sys.path.remove(str(parent))
    except ValueError: # Already removed
        pass
    import wFunction
    __package__ = 'wFunction'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    #fiddling done
    from . import compress_algs as calgs
    from . import interpolate as terp
    from . import  Generate_circuit as gerc
    import matplotlib.pyplot as plt
    import seaborn as sb
    sb.set_theme()
    # N = 6
    # X = qtt.networks.random_MPS(N,4,2)
    # qX = qttMPS2quimbTN(X,'lfq').compress_all_().compress_all_()
    # O = TwoqbitsLosange(qX)
    # print(O)
    # O.draw()
    def f(x):
        return np.exp(-(x)**2)
    from scipy.stats import lognorm as scplog
    def lognorm(x,mu,sigma):
        return scplog.pdf(np.exp(-mu)*x,sigma )
    f = lambda x : lognorm(x,1,1)
    nqbit = 3
    zero_state = trivial_state(nqbit)
    domain = (0,8)
    X = np.linspace(domain[0],domain[1],1000)
    w = np.linspace(*domain,2**nqbit)
    precision = 0.00001
    Gate_precision = precision#1e-12
    MPS_precision = precision #0.001
    polys = gerc.poly_by_part(f,MPS_precision,nqbit,domain)
    plt.plot(X,f(X))
    for poly,subdomain in polys:
        subdo = (terp.bits2range(subdomain[0],domain,nqbit),terp.bits2range(subdomain[1],domain,nqbit))
        ww = np.linspace(*subdo,300)
        plt.plot(ww,poly(ww), label = "poly {} {}".format(subdo,subdomain))
    plt.legend()
    plt.show()
    target_norm2 = 0
    mpses = [
        terp.polynomial2MPS(poly,nqbit,pdomain,domain)
        for poly,pdomain in polys]
    for mps in mpses:
        s = []
        for binstate in range(2**nqbit):
            M = mps
            ts = trivial_state(nqbit,'lfq',binstate)
            s.append(M@ts)
        plt.plot(w,s,label = "mps")
    plt.legend()
    plt.show()
    for mps in mpses:
        target_norm2 += mps@mps
    X = calgs.MPS_compressing_sum(mpses,target_norm2,0.1*MPS_precision,MPS_precision)
    # print_sizes(X)
    Norm = np.sqrt(X@X)
    print("NORM:",Norm)
    X /= Norm
    qX = X
    SM = []
    for binstate in range(2**nqbit):
        ts = trivial_state(nqbit,'lfq',binstate)
        SM.append(qX@ts*Norm)
    # plt.plot(w,SM)
    # plt.show()
    pass
    O = TwoqbitsStaircaseLayers(qX,1)
    # O.draw()
    O, inf = MPS2Gates(X,Gate_precision,1,max_count=10)
    # O.draw(show_tags=True, show_inds=True)
    # O = optimize_unitaries(O,"O\d+",zero_state,qX,Gate_precision,max_iteration=100)
    # for t in qX:
    #     print(t)
    SO = []
    SM = []
    # for t in O:
    #     print(t)
    # print(zero_state)
    for binstate in range(2**nqbit):
        ts = trivial_state(nqbit,'lfq',binstate)
        Val = (zero_state&O)@ts
        SO.append(Val)
        SM.append(qX@ts)
    # print(SO)
    # print(SM)
    SO = np.array(SO)
    SM = np.array(SM)
    print( "expected state vector" ,SM)
    print( "expected operators (no permute): ")
    for t in O:
        print(t.tags)
        print(t.data.reshape(4,4))
    # print("<0|O^d O |0>",(zero_state&O)@(zero_state&O))
    # print(qX@qX)
    # F = (zero_state&O)@qX
    # print(F)
    # print(np.dot(SO,SM))
    # print((1-F)**2)
    plt.plot(SO)
    plt.plot(SM)
    plt.show()



    