from jax.config import config
config.update("jax_enable_x64", True)

import quimb as qb
import quimb.tensor as qtn
# import quantit as qtt
import numpy as np
import jax.numpy as jnp
from quimb.tensor.optimize import TNOptimizer
from .mps2qbitsgates import generate_staircase_operators,unitarity_cost2,gauge_regularizer,generate_Lagrange_multipliers,normalize_gates
from .Chebyshev import pad_reshape

qtn.MatrixProductState
TN = qtn.TensorNetwork

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



def TwoqbitsLayers(mpo:qtn.MatrixProductOperator,Nlayer:int,dtype=jnp.float64,layerGenerator = generate_staircase_operators):
    L = (mpo.L) #number of qbits
    out = qtn.TensorNetwork([])
    lower_ind_id = mpo.upper_ind_id
    upper_ind_id = mpo.lower_ind_id
    for layer in range(0,Nlayer-1 ):
        out &= layerGenerator(lower_ind_id,"l{}q{}".format(layer+1,"{}"),L-1,layer)
        lower_ind_id = "l{}q{}".format(layer+1,"{}")
    out &= layerGenerator(lower_ind_id,upper_ind_id,L-1,Nlayer-1)
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
    Pzero = (Net|brazero)@ketzero
    # Pnorm = Pzero@Pzero.H
    return jnp.real( Pzero.H@Pzero - Pzero.H@mpo )

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

def projection_loss(circuit,mpo,L,id,C,m):
    return (projection_infidelity(circuit,mpo) + unitarity_cost2(circuit,L,id))*(1 + gauge_regularizer(circuit,id,C,m))

def full_loss(circuit,mpo,L,id,C,m,id_net):
    return (Infidelity(circuit,mpo,id_net) + unitarity_cost2(circuit,L,id))*(1 + gauge_regularizer(circuit,id,C,m))
    

def gen_id_net(mpo:qtn.MatrixProductOperator,factor=1):
    dat = [ np.array([[np.eye(2)*factor]]) for i in range(mpo.L)]
    dat[0] = dat[0][0,:,:,:]
    dat[-1] = dat[0]
    return qtn.MatrixProductOperator(dat,upper_ind_id=mpo.upper_ind_id,lower_ind_id=mpo.lower_ind_id)

def stairase2MPO(layer:qtn.TensorNetwork):
    """convert a staircase layer to a MPO."""
    pass



def Proj_MPSO2Gates(mpo:qtn.MatrixProductOperator,precision,Nlayer,max_count=40):
    """Assume the mpo orthogonality center is tensor 0."""
    mpo = mpo.copy()
    N_fact = np.sqrt(mpo[0]@mpo[0].H)
    for T in mpo.tensors: 
        T /= N_fact
    I = np.array([[1.,0.],[0.,1.]])
    O_gen_mpo = qtn.MatrixProductOperator([ *[x.data for x in mpo.tensors[:-1]],pad_reshape(mpo.tensors[-1].data),I.reshape(1,*I.shape) ],site_tag_id=mpo.site_tag_id,upper_ind_id=mpo.upper_ind_id,lower_ind_id=mpo.lower_ind_id)
    O = TwoqbitsLayers(O_gen_mpo,Nlayer,dtype=jnp.complex128)
    L = generate_Lagrange_multipliers(O)
    # print("O",O)
    # print("X",qX)
    # print("ts",ts)
    print(projection_loss(O,mpo,N_fact))
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
        error = projection_loss(OL_opt,mpo,N_fact)
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
        loss_kwargs = {'C':0.1,'m':50},
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
