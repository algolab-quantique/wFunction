from jax.config import config
config.update("jax_enable_x64", True)

from typing import Optional
import quimb as qb
import quimb.tensor as qtn
# import quantit as qtt
import numpy as np
import jax.numpy as jnp
from quimb.tensor.optimize import TNOptimizer
from .mps2qbitsgates import generate_staircase_operators,unitarity_cost2,gauge_regularizer,generate_Lagrange_multipliers,normalize_gates,staircaselayer
from .Chebyshev import pad_reshape
from .MPO_compress import MPO_compressor
from multimethod import multimethod,overload
qtn.MatrixProductState
TN = qtn.TensorNetwork


class V_layer():
    def __init__(self,data = np.random.rand(4,4),uuid = qtn.rand_uuid() + '{}'):
        self.data = data.reshape(2,2,2,2)
        self.data /= np.sum(self.data**2)
        self.uuid = uuid
    def __call__(self,input_idx, output_idx,Nlink,dtype=jnp.float64) -> qtn.TensorNetwork:
        out = qtn.TensorNetwork([])
        Linner = self.uuid.format('L') + '{}'
        Rinner = self.uuid.format('R') + '{}'
        Xinner = self.uuid.format('X') + '{}'
        i = Nlink-1
        LR = i
        if Nlink > 1:
            out &= qtn.Tensor(data =jnp.copy(self.data), inds = [Linner.format(i),input_idx.format(i+1),Rinner.format(i),output_idx.format(i+1)],tags=['L{}'.format(i),"ML{}".format(i),'ML'])
            for i in range(Nlink-2,0,-1):
                LR += 1
                out &= qtn.Tensor(data =jnp.copy(self.data), inds = [Linner.format(i),input_idx.format(i+1),Xinner.format(i),Linner.format(i+1)],tags=['L{}'.format(i),"ML{}".format(i),'ML'])
                out &= qtn.Tensor(data =jnp.copy(self.data), inds = [Xinner.format(i),Rinner.format(i+1),output_idx.format(i+1),Rinner.format(i)],tags=['L{}'.format(LR),"MR{}".format(i),'MR'])
            i = 0
            LR+=1
            out &= qtn.Tensor(data =jnp.copy(self.data), inds = [input_idx.format(i),input_idx.format(i+1),Xinner.format(i),Linner.format(i+1)],tags=['L{}'.format(i),"ML{}".format(i),'ML'])
            out &= qtn.Tensor(data =jnp.copy(self.data), inds = [Xinner.format(i),Rinner.format(i+1),output_idx.format(i+1),output_idx.format(i)],tags=['L{}'.format(LR),"MR{}".format(i),'MR'])
        else:
            out &= qtn.Tensor(data =jnp.copy(self.data), inds = [input_idx.format(i),input_idx.format(i+1),output_idx.format(i),output_idx.format(i+1)],tags=['L{}'.format(i),"ML{}".format(i)])
        return out

def Vlayer2MPO(layer:qtn.TensorNetwork,input_idx:str,output_idx:str):
    """convert a staircase layer to a MPO."""
    Out_tens = []
    Ntens = len(layer.tensors)
    Nlink = (Ntens+1)//2
    X = 'L{}'
    L = 'ML{}'
    R = 'MR{}' 
    t = layer[X.format(Nlink-1)]
    Nr = layer[R.format(Nlink-2)]
    Nl = layer[L.format(Nlink-2)]
    indis = set(t.inds).difference(Nr.inds).difference(Nl.inds)
    tl,tr = t.split(left_inds=None,right_inds = indis,absorb='both',cutoff=0.0)
    tr.transpose_(tr.inds[0],input_idx.format(Nlink),output_idx.format(Nlink))
    Out_tens.append(tr)
    Out_tens.append(tl)
    for f in range(Nlink-2,0,-1):
        t = (Out_tens[-1]|Nr)@Nl
        Nr = layer[R.format(f-1)]
        Nl = layer[L.format(f-1)]
        indis = set(t.inds).difference(Nr.inds).difference(Nl.inds)
        tl,tr = t.split(left_inds=None,right_inds = indis,absorb='both',cutoff=1e-16)
        tr.transpose_(tr.inds[0],tr.inds[-1],input_idx.format(Nlink),output_idx.format(Nlink))
        Out_tens[-1] = tr
        Out_tens.append(tl)
    ins = Out_tens[-1].inds
    Out_tens[-1].transose(ins[-1],input_idx.format(0),output_idx.format(0))
    return qtn.MatrixProductOperator([t.data for t in reversed(Out_tens)],shape = 'lrdu',upper_ind_id=output_idx,lower_ind_id=input_idx)    

class stacklayer:

    def __init__(self,stack_ctrl = -1,data_gen = jnp.eye(4,4)):
        self.stack_control = -1
        self.data_gen = data_gen.reshape(2,2,2,2)
        self.data_gen /= np.sum(self.data_gen**2)
        

    def stackLayer(self,input_idx, output_idx,Nlink, min_layer_number,dtype=jnp.float64):
        out = qtn.TensorNetwork([])
        i = 0
        #prepare l'indice de control pour la taille du problème présent
        control = self.stack_control
        while control < 0:
            control += Nlink+1
        control %= (Nlink+1)
        if control == 0:
            start = Nlink
            step = -1
            stop = 0
        else:
            start = 0
            step = 1
            stop = Nlink - 1
        uuid = qtn.rand_uuid()+"{}"
        left_in_ind = input_idx
        control_in_ind = input_idx.format(control)
        left_out_ind = output_idx
        if Nlink > 1:
            for i in range(start,stop,step):
                if i == control:
                    continue
                c_out_ind = uuid.format(i)
                out &= qtn.Tensor(data = jnp.copy(self.data_gen), inds = [left_in_ind.format(i),control_in_ind,left_out_ind.format(i),c_out_ind],tags=['O','L{}'.format(i+min_layer_number),"O{},{}".format(i+min_layer_number,i)])
                control_in_ind = c_out_ind
            i = stop
            if control != i:
                out &= qtn.Tensor(data = jnp.copy(self.data_gen), inds = [left_in_ind.format(i),control_in_ind,output_idx.format(i),output_idx.format(control)],tags=['O','L{}'.format(i+min_layer_number),"O{},{}".format(i+min_layer_number,i)])
        else:
            out &= qtn.Tensor(data =jnp.copy(self.data_gen), inds = [input_idx.format(start),input_idx.format(control),output_idx.format(start),output_idx.format(control)],tags=['O','L{}'.format(i+min_layer_number),"O{},{}".format(i+min_layer_number,i)])
        return out

    def __call__(self,input_idx, output_idx,Nlink, min_layer_number,dtype=jnp.float64):
        return self.stackLayer(input_idx, output_idx,Nlink, min_layer_number,dtype=dtype) 


class layer_compose():
    def __init__(self,*layers):
        self.layers = layers
    
    def __call__(self,input_idx, output_idx,Nlink, min_layer_number,dtype=jnp.float64):
        n_layer = len(self.layers)
        L_in_idx = input_idx
        uuid = qtn.rand_uuid() + '{}'
        out = qtn.TensorNetwork([])
        for i in range(n_layer-1):
            L_out_idx = uuid.format(i) + '{}'
            out &= self.layers[i](L_in_idx,L_out_idx,Nlink,min_layer_number,dtype)
            L_in_idx = L_out_idx
        out &= self.layers[-1](L_in_idx,output_idx,Nlink,min_layer_number,dtype)
        return out


@overload
def TwoqbitsLayers(mpo:qtn.MatrixProductOperator,Nlayer:int,dtype=jnp.float64,layerGenerator = generate_staircase_operators):
    L = (mpo.L) #number of qbits
    lower_ind_id = mpo.upper_ind_id
    upper_ind_id = mpo.lower_ind_id
    return TwoqbitsLayers(L,lower_ind_id,upper_ind_id,Nlayer,dtype,layerGenerator)

@overload
def TwoqbitsLayers(Nsites:int, lower_ind_id:str, upper_ind_id:str,Nlayer:int,dtype=jnp.float64,layerGenerator=generate_staircase_operators):
    out = qtn.TensorNetwork([])
    for layer in range(0,Nlayer-1 ):
        out &= layerGenerator(lower_ind_id,"l{}q{}".format(layer+1,"{}"),Nsites-1,layer)
        lower_ind_id = "l{}q{}".format(layer+1,"{}")
    out &= layerGenerator(lower_ind_id,upper_ind_id,Nsites-1,Nlayer-1)
    return out

import quimb

def projection_infidelity(Onet:qtn.TensorNetwork,mpo:qtn.MatrixProductOperator,N_fact:float):
    """Sidesteps the representation problem of sqrt(1-f) by projecting the unitary such that it extract f.
    sqrt(1-f) naturally will appear in the part unconstrained by this cost function by enforcing unitarity.
    The trace of the square MPO must be 1 and the target norm is N_fact^nqbit for the loss function to coincide with an infidelity measure. In any case, it must be of low magnitude <3."""
    n = mpo.L
    ketzero = qtn.Tensor(data = [1,0],inds=(mpo.lower_ind_id.format(n),) )
    brazero = qtn.Tensor(data = [1,0],inds=(mpo.upper_ind_id.format(n),) )
    Net = Onet.copy()
    for t in Net.tensors[0:n]:
        t/=N_fact
    Pzero = (Net|brazero)|ketzero
    # Pnorm = Pzero@Pzero.H
    return jnp.real( Pzero.H@Pzero - 2*Pzero.H@mpo + mpo.H@mpo)

def Infidelity(Onet,mpo:qtn.MatrixProductOperator,id_net:qtn.MatrixProductOperator):
    # id_net.upper_ind_id = mpo.upper_ind_id
    # mpo.draw()
    # Onet.draw()
    # id_net.draw()
    O = Onet@mpo.H
    # O2 = O@O
    OO = Onet.copy()
    uid = qtn.rand_uuid()+'{}'
    inds = [uid.format(i) for i,x in enumerate(id_net)]
    OO = OO.reindex( {id:ind for id,ind in zip(id_net.upper_inds,inds)})
    idid = id_net.copy()
    idid.lower_ind_id = uid
    Os = OO|idid
    OOdag = Os@Os.H
    E = 1-2*jnp.real(O)+OOdag
    return E

def projection_loss(circuit,mpo,L,id,C,m,N_fact):
    return (projection_infidelity(circuit,mpo,N_fact) + unitarity_cost2(circuit,L,id))*(1 + gauge_regularizer(circuit,id,C,m))

def full_loss(circuit,mpo,L,id,C,m,id_net):
    return (Infidelity(circuit,mpo,id_net) + unitarity_cost2(circuit,L,id))*(1 + gauge_regularizer(circuit,id,C,m))
    

def gen_id_net(mpo:qtn.MatrixProductOperator,factor=1):
    dat = [ np.array([[np.eye(2)*factor]]) for i in range(mpo.L)]
    dat[0] = dat[0][0,:,:,:]
    dat[-1] = dat[0]
    return qtn.MatrixProductOperator(dat,upper_ind_id=mpo.upper_ind_id,lower_ind_id=mpo.lower_ind_id)

def staircase2MPO(layer:qtn.TensorNetwork,input_idx:str,output_idx:str,bond_idx:str,single_tensor_tag):
    """convert a staircase layer to a MPO."""
    Out_tens = []
    L = single_tensor_tag
    t = layer[L.format(0)]
    tl,tr = t.split(left_inds=[input_idx.format(0),output_idx.format(0)],absorb='both',cutoff=0.0)
    inds = [ tl.inds[-1], *tl.inds[:-1]]
    tl.transpose_(*inds)
    Out_tens.append(tl)
    Out_tens.append(tr)
    for f,_ in enumerate(layer.tensors[1:]):
        i = f+1
        t = layer[L.format(i)]
        tl,tr = t.split(left_inds=[bond_idx.format(i),output_idx.format(i)],absorb='both',cutoff=0.0)
        Out_tens[-1] = Out_tens[-1]@tl
        inds = [Out_tens[-1].inds[0],Out_tens[-1].inds[-1],*Out_tens[-1].inds[1:-1]]
        Out_tens[-1].transpose_(*inds)
        Out_tens.append(tr)
    return qtn.MatrixProductOperator([t.data for t in Out_tens],shape = 'lrdu',upper_ind_id=output_idx,lower_ind_id=input_idx)   #À vérifier. 


def iterative_projection_infidelity(new_layer:qtn.TensorNetwork,MPO_accumulator:qtn.MatrixProductOperator,target_mpo:qtn.MatrixProductOperator):
    """compute $\left|\left| O - \mathrm{Tr}_b(\ket{0_b}\bra{0_b}\mathbf{C}_{n-1}\mathbf{L}_n) \right|\right|^2 \\
  &+ \sum_l \left|\left| \mathbb{1} - \mathbf{c}_l^\dagger\mathbf{c}_l\right|\right|^2$ with $C_n-1 = $ MPO_accumulator, $O = $ target_mpo and $L_n = $ new_layer. """
    n = target_mpo.L
    ketzero = qtn.Tensor(data = [1,0],inds=(target_mpo.lower_ind_id.format(n),) )
    brazero = qtn.Tensor(data = [1,0],inds=(target_mpo.upper_ind_id.format(n),) )
    OA = (MPO_accumulator|new_layer)|( ketzero | brazero)
    beuh = brazero|MPO_accumulator|ketzero
    # print("bra accu ket norm", beuh.H@beuh)
    # print("accu norm", MPO_accumulator.H@MPO_accumulator)
    # print("layer norm", new_layer.H@new_layer)
    # print("circuit norm",OA.H@OA)
    # print("scalar prod", OA.H@target_mpo)
    return jnp.real( OA.H@OA + target_mpo.H@target_mpo - 2 * OA@target_mpo.H )

def generate_id_MPO(upper_ind:str,lower_ind:str,Nsite:int,factor:float = 1):
    id = [ np.array([[ np.eye(2)*factor]]) for i in range(Nsite)]
    id[0] = id[0][0,:,:,:]
    id[-1] = id[0]
    return qtn.MatrixProductOperator(id,upper_ind_id=upper_ind,lower_ind_id=lower_ind)

def generate_swap_MPO(upper_ind:str,lower_ind:str,Nsite:int,swap_sites:list[int] = [0,-1]):
    if swap_sites[1] < 0:
        swap_sites[1] += Nsite
    if swap_sites[0] < 0:
        swap_sites[0] += Nsite
    id = np.array([[ np.eye(2)]])
    transmit =  np.eye(8,8).reshape(4,2,4,2).transpose(0,2,1,3)
    X,iY,Z,I = np.array([[0,1],[1,0]]),np.array([[0,1],[-1,0]]),np.array([[1,0],[0,-1]]),np.array([[1,0],[0,1]])
    Lswap = np.zeros((1,4,2,2))
    Lswap[0,0,:,:] = iY/2
    Lswap[0,1,:,:] = X/2
    Lswap[0,2,:,:] = Z/2
    Lswap[0,3,:,:] = I/2
    Rswap = np.zeros((4,1,2,2))
    Rswap[0,0,:,:] = -iY/2
    Rswap[1,0,:,:] = X
    Rswap[2,0,:,:] = Z
    Rswap[3,0,:,:] = I
    def select(i):
        if i < swap_sites[0]:
            return id
        if i == swap_sites[0]:
            return Lswap
        if i < swap_sites[1]:
            return transmit
        if i == swap_sites[1]:
            return Rswap
        return id
    op = [select(i) for i in range(Nsite)]
    op[0] = op[0][0,:,:,:]
    op[-1] = op[-1][:,0,:,:]
    return qtn.MatrixProductOperator(op,upper_ind_id=upper_ind,lower_ind_id=lower_ind)

    

def retag(X:qtn.Tensor,tags):
    X.drop_tags()
    for tag in tags: 
        X.add_tag(tag)

def layer_SVD_extractBx(layer,accu,id,currpos,tmp_id):
    Bp = (layer|accu|id)
    Breinds = set(layer[currpos].inds).difference(Bp.outer_inds())
    Breinds = { t:tmp_id.format(i) for i,t in enumerate(Breinds)}
    # assert(len(Breinds) == 3)
    BpH = Bp.H.reindex(Breinds)
    curtag = BpH[currpos].tags
    BpH[currpos].add_tag("around_this")
    BpH[currpos].drop_tags(curtag)#drop all tags but "around_this"
    B = (Bp|BpH)
    BLtags = set(B.tags).difference(set(layer[currpos].tags).union(["around_this"]))
    B = B.contract(BLtags) #contract all tensor but the one with "around_this" tag
    Bc = B.contract()
    # assert(len(B.tensors) == 3)
    B = B[BLtags]

    return B,layer[currpos],Breinds,Bc

def layer_SVD_extractH(layer,accu,mpo,id,currpos):
    T = layer|accu|mpo.H|id
    t = T.tags
    t = t.difference(T[currpos].tags)
    Tc = T.contract(t)
    H = Tc[t]
    Tc = Tc.contract()
    return H,Tc



def layer_SVD_optimizer(layer:qtn.TensorNetwork,accu:qtn.MatrixProductOperator,mpo:qtn.MatrixProductOperator,Layer_tag_id:str,max_it:int,tgt_prec:int):
    """The tags contained in layer must be unique: each tensor of layer is uniquely identified by it, and those tags are absent from accu and mpo.
    Each tensor of the layer must be tagged by Layer_tag_id.format{i} where i is in range(len(later.tensors))
    The current layer+initial MPO topology isn't able to converge to a solution.
    Other topologies means the local problem is trickier.
    """
    n = mpo.L
    layer = layer.copy()
    idzero = qtn.Tensor(data = [[1,0],[0,0]],inds=(mpo.lower_ind_id.format(n),mpo.upper_ind_id.format(n),) ,tags = ["IDzero"])
    idone = qtn.Tensor(data = [[0,0],[0,1]],inds=(mpo.lower_ind_id.format(n),mpo.upper_ind_id.format(n),) ,tags = ["IDone"])
    ket = qtn.Tensor(data = [1,0],inds=(mpo.lower_ind_id.format(n),) ,tags = ["ket"])
    bra = qtn.Tensor(data = [1,0],inds=(mpo.upper_ind_id.format(n),) ,tags = ["bra"])
    Ntens = len(layer.tensors)
    layertags = [Layer_tag_id.format(i) for i in range(Ntens)]
    err = 1000
    it = 0
    direction = 1
    pos = 0
    tmp_id = qtn.rand_uuid() + '{}'
    tgt_norm = mpo@mpo.H
    Ltags = [Layer_tag_id.format(pos) for pos in range(Ntens)]
    N = mpo@mpo.H
    while it<max_it and err > tgt_prec:
        it+=1
        print("direction ",direction)
        for i in range(Ntens):
            curr_A = Ltags[pos]
            Bzero,l,Breinds,Bc = layer_SVD_extractBx(layer,accu,idzero,curr_A,tmp_id)
            Bone,l,Breinds,Bc = layer_SVD_extractBx(layer,accu,idone,curr_A,tmp_id)
            Hzero,Tc = layer_SVD_extractH(layer,accu,mpo,idzero,curr_A)
            Hone,Tc = layer_SVD_extractH(layer,accu,mpo,idone,curr_A)
            currerr = np.real(Bc-2*Tc+N)
            print("current error :", currerr)
            Hzero.reindex_(Breinds)
            Hone.reindex_(Breinds)
            if pos == 0:
                oli = set(layer[curr_A].inds).intersection(accu[pos].inds).union(set(layer[curr_A].inds).intersection(accu[pos+1].inds))
            else:
                oli = set(layer[curr_A].inds).intersection(layer[Ltags[pos-1]].inds).union(set(layer[curr_A].inds).intersection(accu[pos+1].inds))
            iri = [ Breinds[i] if i in Breinds else i for i in oli ]
            #hypothesis, first two indices are input, last two are output. Or vice versa
            upd = biupd(Bzero,Hzero,Bone,Hone,layer[curr_A],oli,iri,N)
            updd = upd.reindex(Breinds).H
            newerr = np.real((Bzero|upd|updd).contract()  + 2*updd@Hzero + N )
            retag(upd,layer[curr_A].tags)
            layer[curr_A] = upd
            # upd,d = layer_upd(B,H,layer[curr_A],oli,iri)
            pos += direction
        ##
        direction *= -1
        pos += direction
        # print(iterative_projection_infidelity(T.select_any(layertags),accu,mpo))
        # print(iterative_projection_infidelity(normalize_gates(T.select_any(layertags)),accu,mpo))
    return layer.select_any(layertags)

def biupd(Bzero:qtn.Tensor,Hzero:qtn.Tensor,Bone:qtn.Tensor,Hone:qtn.Tensor,l:qtn.Tensor,oli:list[str],iri:list[str],tnorm:float=0) -> qtn.Tensor:
    tmp_inds = set(Hzero.inds).intersection(Bzero.inds)
    Ub0,sb0,Vb0 = Bzero.split(None,right_inds = tmp_inds,absorb = None, cutoff = 1e-13)
    Ub1,sb1,Vb1 = Bone.split(None,right_inds = tmp_inds,absorb = None, cutoff = 1e-13)

    sb0m = 1/sb0
    sb1m = 1/sb1
    A0 = (Ub0.H|sb0m|Vb0.H|Hzero).contract(output_inds = l.inds)
    A1 = (Ub1.H|sb1m|Vb1.H|Hone).contract(output_inds = l.inds)
    Ac = (A0+A1)#0.5? sans importance: normlize élimine ce genre de facteur.
    l = Ac 
    # def cost(l0,Ac):
    #     return l0@l0.H + Ac@Ac.H - l0@Ac.H + Ac@l0.H
    def normalize(l0):
        u,d,v = l0.split(oli,cutoff = 0, absorb=None)
        return qtn.TensorNetwork( [u@v] )
    # opt = qtn.TNOptimizer(qtn.TensorNetwork([l]),cost,normalize,{'Ac':qtn.TensorNetwork([Ac])},loss_target=0.01,progbar=False)
    # l = opt.optimize(10)
    return normalize(l).tensors[0]
    
def layer_upd(B:qtn.Tensor,H:qtn.Tensor,A0:qtn.Tensor, output_left_inds:list[str],input_right_inds:list[str]):
    """
    compute H_{a,b,c,d}B^{-1}_{e,f,g,b,c,d}
    singularities of B are removed with the constraint that the output must be unitary
    Initiallement je vais operer sous l'hypothese que H.B^{-1} a seulement 2 valeur singulière non nul.
    """
    tmp_inds = set(H.inds).intersection(B.inds)
    output_right_ind = set(A0.inds).difference(output_left_inds)
    Ub,sb,Vb = B.split(None,right_inds = tmp_inds,absorb = None, cutoff = 1e-13)
    sbm = 1/sb
    A = (Ub.H|sbm|Vb.H|H).contract(output_inds = A0.inds)
    Ua,sa,Va = A.split(output_left_inds,absorb = None,cutoff = 0)
    # return Ua@Va,sa
    if sa.data[0] > 1:
        sa /= sa.data[0]
    if len( sa.data[sa.data>1e-14]) > 2 :
        return Ua@Va,sa
        #woopsie
    else:
        indt = qtn.rand_uuid()
        d = (sa.data[sa.data > 1e-14])
        dt = np.diag(np.sqrt(1-d**2))
        d = np.diag(d)
        At = np.zeros((4,4))
        At[:2,:2] = d
        At[2:,2:] = d
        At[2:,:2] = dt
        At[:2,2:] = -dt
        Ua.reindex_({sa.inds[0]:indt})
        At = qtn.Tensor(At,inds = [indt,sa.inds[0]])
        return (Ua|At|Va).contract(),sa
    


def Iterative_MPOembedGatesA(mpo:qtn.MatrixProductOperator,precision:float,maxlayers:int=40):
    N_fact = np.sqrt(mpo[0]@mpo[0].H)#this assumes a canonical MPO: each and every tensor contribute the same value to the total norm^2 of the network, and 0 is the center.
    layercount = 0
    uid = qtn.rand_uuid() + '{}'
    layer_link_id = qtn.rand_uuid() + '{}'
    left_llid = layer_link_id.format('l{}')
    right_llid = layer_link_id.format('r{}')
    mpo = mpo.copy()
    Ltag = 'L'
    Rtag = 'R'
    Lop_tag = Ltag+'{}'
    Rop_tag = Rtag+'{}'
    accuhigh = uid.format(layercount)+'h{}'
    acculow = uid.format(layercount)+'l{}'
    accumulator = generate_id_MPO(accuhigh ,acculow,mpo.L+1,1)
    circuit = qtn.TensorNetwork([])

    n = mpo.L
    projzero = qtn.Tensor(data = [[1,0],[0,0]],inds=(mpo.lower_ind_id.format(n),mpo.upper_ind_id.format(n),) ,tags = ["IDzero"])
    
    right_layer_gen = staircaselayer(uuid=right_llid,data = np.eye(4,4))
    right_layer_gen.data = np.eye(4,4).reshape(2,2,2,2) ##because there's a normalisation happening that is undesirable in this case.
    left_layer_gen = staircaselayer(uuid=left_llid,data = np.eye(4,4))
    left_layer_gen.data = np.eye(4,4).reshape(2,2,2,2) ##idem
    accumulator = generate_id_MPO(accuhigh ,acculow,mpo.L+1,1)
    right_layer = right_layer_gen(mpo.upper_ind_id,accuhigh,mpo.L,0,op_tag=Rop_tag,layer_tag=Rtag)
    left_layer = left_layer_gen(mpo.lower_ind_id,acculow,mpo.L,0,op_tag=Lop_tag,layer_tag=Ltag)

    error = 1000
    output_map = {}
    for i in range(mpo.L+1):
        output_map[accuhigh.format(i)] = acculow.format(i)
    Layer_ind_id = 'LR{}'
    while error>precision and layercount < maxlayers:
        layer = left_layer|right_layer
        for i,t in enumerate(layer):
            t.add_tag(Layer_ind_id.format(i))
        ##
        if layercount == 0:
            (layer|accumulator|projzero).draw()
            LL = layer.reindex(output_map)
            (LL|projzero).draw(initial_layout='spiral')
        ##
        layer = layer_SVD_optimizer(layer,accumulator,mpo,Layer_ind_id,10,1e-4) #Current behavior is anomalous. we're using the same function, so it's the layer&accumulator combo that is at fault.
        layercount += 1
        if (error>precision and layercount < maxlayers):
            next_accuhigh = uid.format(layercount)+'h{}'
            next_acculow = uid.format(layercount)+'l{}'
            rein_map = {}
            for i in range(mpo.L+1):
                rein_map[mpo.upper_ind_id.format(i)] = next_accuhigh.format(i)
                rein_map[mpo.lower_ind_id.format(i)] = next_acculow.format(i)
            layer.reindex_(rein_map)
        else:
            next_accuhigh = mpo.upper_ind_id
            next_acculow = mpo.lower_ind_id

        left_layer = qtn.TensorNetwork(layer['L'])
        right_layer = qtn.TensorNetwork(layer['R'])
        mpo_left_layer = staircase2MPO(left_layer,acculow,next_acculow,left_llid,Lop_tag)#the indices are wrong
        mpo_right_layer = staircase2MPO(right_layer,accuhigh,next_accuhigh,right_llid,Rop_tag)
        # print("mpo@L", mpo_layer.H@layer)
        circuit &= layer
        accuhigh = next_accuhigh
        acculow = next_acculow
        right_layer = right_layer_gen(accuhigh,mpo.upper_ind_id,mpo.L,0,op_tag=Rop_tag,layer_tag=Rtag)
        left_layer = left_layer_gen(mpo.lower_ind_id,acculow,mpo.L,0,op_tag=Lop_tag,layer_tag=Ltag)

        accumulator.site_tag_id = "I{}" # tag sanitization, because it seems .apply rely on this, but doesn't check.
        mpo_right_layer.site_tag_id = "I{}"
        mpo_left_layer.site_tag_id = "I{}"
        accumulator = mpo_right_layer.apply( accumulator.apply(mpo_left_layer))
        accumulator.upper_ind_id = "__TMP__{}"
        accumulator.lower_ind_id = acculow
        accumulator.upper_ind_id = accuhigh
        ####t'es rendu ici.
        accumulator = MPO_compressor(accumulator,precision*0.01,precision*0.02)
        # layer.draw()
        # accumulator.draw()
    # for i in range(mpo.L+1):
    #     output_map[acculow.format(i)] = mpo.lower_ind_id.format(i)
    #     output_map[accuhigh.format(i)] = mpo.upper_ind_id.format(i)
    
    accumulator.upper_ind_id = mpo.upper_ind_id
    accumulator.lower_ind_id = mpo.lower_ind_id

    circuit.reindex_(output_map)
    return circuit,accumulator

        
    

def Proj_MPSO2Gates(mpo:qtn.MatrixProductOperator,precision,Nlayer,max_count=40,layer = generate_staircase_operators):
    """Assume the mpo orthogonality center is tensor 0."""
    mpo = mpo.copy()
    N_fact = np.sqrt(mpo[0]@mpo[0].H)
    for T in mpo.tensors: 
        T /= N_fact
    print("tgt norm", mpo@mpo.H)
    I = np.array([[1.,0.],[0.,1.]])
    O_gen_mpo = qtn.MatrixProductOperator([ *[x.data for x in mpo.tensors[:-1]],pad_reshape(mpo.tensors[-1].data),I.reshape(1,*I.shape) ],site_tag_id=mpo.site_tag_id,upper_ind_id=mpo.upper_ind_id,lower_ind_id=mpo.lower_ind_id)
    O = TwoqbitsLayers(mpo,Nlayer,layerGenerator=layer)
    O = TwoqbitsLayers(O_gen_mpo,Nlayer,dtype=jnp.complex128)
    L = generate_Lagrange_multipliers(O)
    # print("O",O)
    # print("X",qX)
    # print("ts",ts)
    print(projection_infidelity(O,mpo,N_fact))
    id = qtn.Tensor(data = jnp.eye(4,4).reshape(2,2,2,2), inds=('a','b','c','d'))
    optmzr = TNOptimizer(
        O,
        loss_fn = projection_loss,
        # norm_fn=normalize_gates,
        loss_kwargs = {'C':0.,'m':50},
        loss_constants={'mpo': mpo,'N_fact':N_fact,'id':id,'L':L},  # this is a constant TN to supply to loss_fn: psi,trivial_state, id, C,m)
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
        error = projection_infidelity(OL_opt,mpo,N_fact)
        # print("count: ", count)
        print("current error: ", error, " unitarity error: ", unitarity_cost2(OL_opt,L,id))
        count += 1 
    O = normalize_gates(OL_opt)
    return O,error

#Doing precisely what i describe in the article will require different definition of loss function and "MPO2Gates"

#Ne scale pas suffisament bien pour être très pratique.
#Il faut optimiser une couche à la fois, composer le résultat à haute précision avec le MPO cible (devrait réduire l'enchevetrement) et calculer la nouvelle couche avec le nouveau MPO ainsi créé.
def MPSO2Gates(mpo:qtn.MatrixProductOperator,precision,Nlayer,max_count=10,layer = generate_staircase_operators):
    nmpo = mpo.copy()
    sqrthalf = np.sqrt(0.5)
    for i,_ in enumerate(mpo):
        nmpo[i] = nmpo[i]*0.5
    mpo = nmpo
    O = TwoqbitsLayers(mpo,Nlayer,layerGenerator=layer)
    L = generate_Lagrange_multipliers(O)
    loss = full_loss    
    # print("O",O)
    # print("X",qX)
    # print("ts",ts)
    id_net = gen_id_net(mpo,sqrthalf)
    print("Initial error",Infidelity(O,mpo,id_net))
    id = qtn.Tensor(data = jnp.eye(4,4).reshape(2,2,2,2), inds=('a','b','c','d'))
    optmzr = TNOptimizer(
        O,
        loss_fn = loss,
        # norm_fn=normalize_gates,
        loss_kwargs = {'C':0.0,'m':50},
        loss_constants={'mpo': mpo, 'id_net':id_net,'id':id,'L':L},  # this is a constant TN to supply to loss_fn: psi,trivial_state, id, C,m)
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
        error = Infidelity(OL_opt,mpo,id_net)
        # print("count: ", count)
        print("current error: ", error, " unitarity error: ", unitarity_cost2(OL_opt,L,id))
        count += 1 
    O = normalize_gates(OL_opt)
    return O,error
# %%
