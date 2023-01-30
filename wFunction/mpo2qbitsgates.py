from jax.config import config

config.update("jax_enable_x64", True)

from typing import Union, Optional
import quimb as qb
import quimb.tensor as qtn

# import quantit as qtt
import numpy as np
import jax.numpy as jnp
from quimb.tensor.optimize import TNOptimizer
from .mps2qbitsgates import (
    generate_staircase_operators,
    unitarity_cost2,
    gauge_regularizer,
    generate_Lagrange_multipliers,
    normalize_gates,
    staircaselayer,
)
from .Chebyshev import pad_reshape
from .generate_simple_MPO import generate_id_MPO
from .MPO_compress import MPO_compressor
from multimethod import multimethod, overload
from . import generate_simple_MPO as gsp
from .compress_algs import layer_SVD_optimizer

qtn.MatrixProductState
TN = qtn.TensorNetwork


class layer_base:
    def __init__(self, data=None, uuid=qtn.rand_uuid() + "{}"):
        if data is None:
            self.data = np.random.rand(2, 2, 2, 2)
            self.data /= np.sum(np.abs(self.data) ** 2)
        else:
            self.data = data
        self.data = self.data.reshape(2, 2, 2, 2)
        self.uuid = uuid

    def __call__(
        self,
        input_idx,
        output_idx,
        Nlink,
        min_layer_number=0,
        dtype=jnp.float64,
        op_tag="L{}",
        layer_tag="L",
        left_right_index=False,
    ) -> Union[qtn.TensorNetwork, tuple[qtn.TensorNetwork, dict, dict]]:
        raise NotImplementedError("called the pure virtual base class")

    @classmethod
    def toMPO(
        cls,
        layer: qtn.TensorNetwork,
        input_idx: str,
        output_idx: str,
        bond_idx: str,
        single_tensor_tag,
    ):
        raise NotImplementedError("called the pure virtual base class")


class brick_layer(layer_base):
    def __call__(
        self,
        input_idx,
        output_idx,
        Nlink,
        min_layer_number=0,
        dtype=jnp.float64,
        op_tag="L{}",
        layer_tag="L",
        left_right_index=False,
    ) -> Union[qtn.TensorNetwork, tuple[qtn.TensorNetwork, dict, dict]]:
        out = qtn.TensorNetwork([])
        inner_idx = self.uuid.format("I") + "{}"
        i = 0
        left_inds = {}
        right_inds = {}
        if Nlink > 1:
            utag = op_tag.format(i)
            left = [input_idx.format(i), input_idx.format(i + 1)]
            right = [output_idx.format(i), inner_idx.format(i + 1)]
            left_inds[utag] = left
            right_inds[utag] = right
            out &= qtn.Tensor(
                jnp.copy(self.data), inds=(*left, *right), tags=(utag, layer_tag)
            )
            for i in range(1, Nlink - 1):
                if i % 2:
                    left = [inner_idx.format(i), inner_idx.format(i + 1)]
                    right = [output_idx.format(i), output_idx.format(i + 1)]
                else:
                    left = [input_idx.format(i), input_idx.format(i + 1)]
                    right = [inner_idx.format(i), inner_idx.format(i + 1)]
                utag = op_tag.format(i)
                left_inds[utag] = left
                right_inds[utag] = right
                out &= qtn.Tensor(
                    jnp.copy(self.data), inds=(*left, *right), tags=(utag, layer_tag)
                )
            i = Nlink - 1
            if Nlink % 2:
                left = [input_idx.format(i), input_idx.format(i + 1)]
                right = [inner_idx.format(i), output_idx.format(i + 1)]
            else:
                left = [inner_idx.format(i), input_idx.format(i + 1)]
                right = [output_idx.format(i), output_idx.format(i + 1)]
            utag = op_tag.format(i)
            left_inds[utag] = left
            right_inds[utag] = right
            out &= qtn.Tensor(
                jnp.copy(self.data), inds=(*left, *right), tags=(utag, layer_tag)
            )
        else:
            utag = op_tag.format(i)
            left = [input_idx.format(i), input_idx.format(i + 1)]
            right = [output_idx.format(i), output_idx.format(i + 1)]
            left_inds[utag] = left
            right_inds[utag] = right
            out &= qtn.Tensor(
                jnp.copy(self.data), inds=(*left, *right), tags=(utag, layer_tag)
            )
        if left_right_index:
            return out, left_inds, right_inds
        else:
            return out

    @staticmethod
    def get_inds_with_i(inds: list[str], i: int):
        out = []
        istr = str(i)
        for ind in inds:
            if ind[-len(istr) :] == istr and not (ind[-len(istr) - 1]).isdigit():
                out.append(ind)
        return out

    @classmethod
    def toMPO(
        cls,
        layer: qtn.TensorNetwork,
        input_idx: str,
        output_idx: str,
        bond_idx: str,
        single_tensor_tag,
    ):
        tensors = []
        i = 0
        T = single_tensor_tag.format(i)
        Tens = layer[T]
        left_inds = cls.get_inds_with_i(Tens.inds, i)
        U, V = Tens.split(left_inds, absorb="left", cutoff=1e-15)
        tensors.append(U)
        tensors[-1] = tensors[-1].data
        tensors.append(V)
        for i in range(1, len(layer.tensors)):
            T = single_tensor_tag.format(i)
            Tens = layer[T]
            left_inds = cls.get_inds_with_i(Tens.inds, i)
            U, V = Tens.split(left_inds, absorb="left", cutoff=1e-15)
            tensors[-1] = (tensors[-1] @ U).data
            tensors.append(V)
        tensors[-1] = tensors[-1].data
        return qtn.MatrixProductOperator(
            tensors, "ldur", lower_ind_id=input_idx, upper_ind_id=output_idx
        )


class V_layer(layer_base):
    def __call__(
        self, input_idx, output_idx, Nlink, dtype=jnp.float64, cpt_inds=False
    ) -> qtn.TensorNetwork:
        out = qtn.TensorNetwork([])
        Linner = self.uuid.format("L") + "{}"
        Rinner = self.uuid.format("R") + "{}"
        Xinner = self.uuid.format("X") + "{}"
        i = Nlink - 1
        LR = i
        left_inds = {}
        right_inds = {}
        if Nlink > 1:
            out &= qtn.Tensor(
                data=jnp.copy(self.data),
                inds=[
                    Linner.format(i),
                    input_idx.format(i + 1),
                    Rinner.format(i),
                    output_idx.format(i + 1),
                ],
                tags=["L{}".format(i), "ML{}".format(i), "ML"],
            )
            left_inds["L{}".format(i)] = (Linner.format(i), input_idx.format(i + 1))
            right_inds["L{}".format(i)] = (Rinner.format(i), output_idx.format(i + 1))
            for i in range(Nlink - 2, 0, -1):
                LR += 1
                out &= qtn.Tensor(
                    data=jnp.copy(self.data),
                    inds=[
                        Linner.format(i),
                        input_idx.format(i + 1),
                        Xinner.format(i),
                        Linner.format(i + 1),
                    ],
                    tags=["L{}".format(i), "ML{}".format(i), "ML"],
                )
                left_inds["L{}".format(i)] = (Linner.format(i), input_idx.format(i + 1))
                right_inds["L{}".format(i)] = (Xinner.format(i), Linner.format(i + 1))
                out &= qtn.Tensor(
                    data=jnp.copy(self.data),
                    inds=[
                        Xinner.format(i),
                        Rinner.format(i + 1),
                        Rinner.format(i),
                        output_idx.format(i + 1),
                    ],
                    tags=["L{}".format(LR), "MR{}".format(i), "MR"],
                )
                left_inds["L{}".format(LR)] = (Xinner.format(i), Rinner.format(i + 1))
                right_inds["L{}".format(LR)] = (
                    Rinner.format(i),
                    output_idx.format(i + 1),
                )
            i = 0
            LR += 1
            out &= qtn.Tensor(
                data=jnp.copy(self.data),
                inds=[
                    input_idx.format(i),
                    input_idx.format(i + 1),
                    Xinner.format(i),
                    Linner.format(i + 1),
                ],
                tags=["L{}".format(i), "ML{}".format(i), "ML"],
            )
            left_inds["L{}".format(i)] = (input_idx.format(i), input_idx.format(i + 1))
            right_inds["L{}".format(i)] = (Xinner.format(i), Linner.format(i + 1))
            out &= qtn.Tensor(
                data=jnp.copy(self.data),
                inds=[
                    Xinner.format(i),
                    Rinner.format(i + 1),
                    output_idx.format(i),
                    output_idx.format(i + 1),
                ],
                tags=["L{}".format(LR), "MR{}".format(i), "MR"],
            )
            left_inds["L{}".format(LR)] = (Xinner.format(i), Rinner.format(i + 1))
            right_inds["L{}".format(LR)] = (
                output_idx.format(i),
                output_idx.format(i + 1),
            )
        else:
            out &= qtn.Tensor(
                data=jnp.copy(self.data),
                inds=[
                    input_idx.format(i),
                    input_idx.format(i + 1),
                    output_idx.format(i),
                    output_idx.format(i + 1),
                ],
                tags=["L{}".format(i), "ML{}".format(i)],
            )
        if cpt_inds:
            return out, left_inds, right_inds
        else:
            return out

    @classmethod
    def toMPO(cls, layer: qtn.TensorNetwork, input_idx: str, output_idx: str):
        """convert a V layer to a MPO."""
        Out_tens = []
        Ntens = len(layer.tensors)
        Nlink = (Ntens + 1) // 2
        X = "L{}"
        L = "ML{}"
        R = "MR{}"
        t = layer[X.format(Nlink - 1)]
        Nr = layer[R.format(Nlink - 2)]
        Nl = layer[L.format(Nlink - 2)]
        indis = set(t.inds).difference(Nr.inds).difference(Nl.inds)
        tl, tr = t.split(left_inds=None, right_inds=indis, absorb="both", cutoff=0.0)
        tr.transpose_(tr.inds[0], input_idx.format(Nlink), output_idx.format(Nlink))
        Out_tens.append(tr)
        Out_tens.append(tl)
        for f in range(Nlink - 2, 0, -1):
            t = (Out_tens[-1] | Nr) @ Nl
            Nr = layer[R.format(f - 1)]
            Nl = layer[L.format(f - 1)]
            indis = set(t.inds).difference(Nr.inds).difference(Nl.inds)
            tl, tr = t.split(
                left_inds=None, right_inds=indis, absorb="both", cutoff=1e-16
            )
            tr.transpose_(
                tr.inds[0],
                tr.inds[-1],
                input_idx.format(Nlink),
                output_idx.format(Nlink),
            )
            Out_tens[-1] = tr
            Out_tens.append(tl)
        ins = Out_tens[-1].inds
        Out_tens[-1].transose(ins[-1], input_idx.format(0), output_idx.format(0))
        return qtn.MatrixProductOperator(
            [t.data for t in reversed(Out_tens)],
            shape="lrdu",
            upper_ind_id=output_idx,
            lower_ind_id=input_idx,
        )


class stacklayer:
    def __init__(self, stack_ctrl=-1, data_gen=jnp.eye(4, 4)):
        self.stack_control = -1
        self.data_gen = data_gen.reshape(2, 2, 2, 2)
        self.data_gen /= np.sum(self.data_gen**2)

    def stackLayer(
        self, input_idx, output_idx, Nlink, min_layer_number, dtype=jnp.float64
    ):
        out = qtn.TensorNetwork([])
        i = 0
        # prepare l'indice de control pour la taille du problème présent
        control = self.stack_control
        while control < 0:
            control += Nlink + 1
        control %= Nlink + 1
        if control == 0:
            start = Nlink
            step = -1
            stop = 0
        else:
            start = 0
            step = 1
            stop = Nlink - 1
        uuid = qtn.rand_uuid() + "{}"
        left_in_ind = input_idx
        control_in_ind = input_idx.format(control)
        left_out_ind = output_idx
        if Nlink > 1:
            for i in range(start, stop, step):
                if i == control:
                    continue
                c_out_ind = uuid.format(i)
                out &= qtn.Tensor(
                    data=jnp.copy(self.data_gen),
                    inds=[
                        left_in_ind.format(i),
                        control_in_ind,
                        left_out_ind.format(i),
                        c_out_ind,
                    ],
                    tags=[
                        "O",
                        "L{}".format(i + min_layer_number),
                        "O{},{}".format(i + min_layer_number, i),
                    ],
                )
                control_in_ind = c_out_ind
            i = stop
            if control != i:
                out &= qtn.Tensor(
                    data=jnp.copy(self.data_gen),
                    inds=[
                        left_in_ind.format(i),
                        control_in_ind,
                        output_idx.format(i),
                        output_idx.format(control),
                    ],
                    tags=[
                        "O",
                        "L{}".format(i + min_layer_number),
                        "O{},{}".format(i + min_layer_number, i),
                    ],
                )
        else:
            out &= qtn.Tensor(
                data=jnp.copy(self.data_gen),
                inds=[
                    input_idx.format(start),
                    input_idx.format(control),
                    output_idx.format(start),
                    output_idx.format(control),
                ],
                tags=[
                    "O",
                    "L{}".format(i + min_layer_number),
                    "O{},{}".format(i + min_layer_number, i),
                ],
            )
        return out

    def __call__(
        self, input_idx, output_idx, Nlink, min_layer_number, dtype=jnp.float64
    ):
        return self.stackLayer(
            input_idx, output_idx, Nlink, min_layer_number, dtype=dtype
        )


class layer_compose:
    def __init__(self, *layers):
        self.layers = layers

    def __call__(
        self, input_idx, output_idx, Nlink, min_layer_number, dtype=jnp.float64
    ):
        n_layer = len(self.layers)
        L_in_idx = input_idx
        uuid = qtn.rand_uuid() + "{}"
        out = qtn.TensorNetwork([])
        for i in range(n_layer - 1):
            L_out_idx = uuid.format(i) + "{}"
            out &= self.layers[i](L_in_idx, L_out_idx, Nlink, min_layer_number, dtype)
            L_in_idx = L_out_idx
        out &= self.layers[-1](L_in_idx, output_idx, Nlink, min_layer_number, dtype)
        return out


@overload
def TwoqbitsLayers(
    mpo: qtn.MatrixProductOperator,
    Nlayer: int,
    dtype=jnp.float64,
    layerGenerator=generate_staircase_operators,
):
    L = mpo.L  # number of qbits
    lower_ind_id = mpo.upper_ind_id
    upper_ind_id = mpo.lower_ind_id
    return TwoqbitsLayers(L, lower_ind_id, upper_ind_id, Nlayer, dtype, layerGenerator)


@overload
def TwoqbitsLayers(
    Nsites: int,
    lower_ind_id: str,
    upper_ind_id: str,
    Nlayer: int,
    dtype=jnp.float64,
    layerGenerator=generate_staircase_operators,
):
    out = qtn.TensorNetwork([])
    for layer in range(0, Nlayer - 1):
        out &= layerGenerator(
            lower_ind_id, "l{}q{}".format(layer + 1, "{}"), Nsites - 1, layer
        )
        lower_ind_id = "l{}q{}".format(layer + 1, "{}")
    out &= layerGenerator(lower_ind_id, upper_ind_id, Nsites - 1, Nlayer - 1)
    return out


def Infidelity(Onet, mpo: qtn.MatrixProductOperator, id_net: qtn.MatrixProductOperator):
    # id_net.upper_ind_id = mpo.upper_ind_id
    # mpo.draw()
    # Onet.draw()
    # id_net.draw()
    O = Onet @ mpo.H
    # O2 = O@O
    OO = Onet.copy()
    uid = qtn.rand_uuid() + "{}"
    inds = [uid.format(i) for i, x in enumerate(id_net)]
    OO = OO.reindex({id: ind for id, ind in zip(id_net.upper_inds, inds)})
    idid = id_net.copy()
    idid.lower_ind_id = uid
    Os = OO | idid
    OOdag = Os @ Os.H
    E = 1 - 2 * jnp.real(O) + OOdag
    return E


def full_loss(circuit, mpo, L, id, C, m, id_net):
    return (Infidelity(circuit, mpo, id_net) + unitarity_cost2(circuit, L, id)) * (
        1 + gauge_regularizer(circuit, id, C, m)
    )


def Iterative_MPOGatesA(
    mpo: qtn.MatrixProductOperator,
    precision: float,
    maxlayers: int = 40,
    layer_gen=staircaselayer,
):
    N_fact = np.sqrt(
        mpo[0] @ mpo[0].H
    )  # this assumes a canonical MPO: each and every tensor contribute the same value to the total norm^2 of the network, and 0 is the center.
    layercount = 0
    uid = qtn.rand_uuid() + "{}"
    layer_link_id = qtn.rand_uuid() + "{}"
    left_llid = layer_link_id.format("l{}")
    right_llid = layer_link_id.format("r{}")
    mpo = mpo.copy()
    Ltag = "L"
    Rtag = "R"
    Lop_tag = Ltag + "{}"
    Rop_tag = Rtag + "{}"
    accuhigh = uid.format(layercount) + "h{}"
    acculow = uid.format(layercount) + "l{}"
    circuit = qtn.TensorNetwork([])

    n = mpo.L

    right_layer_gen = layer_gen(uuid=right_llid)
    right_layer_gen.data = np.eye(4, 4).reshape(
        2, 2, 2, 2
    )  ##because there's a normalisation happening that is undesirable in this case.
    left_layer_gen = layer_gen(uuid=left_llid)
    left_layer_gen.data = np.eye(4, 4).reshape(2, 2, 2, 2)  ##idem
    accumulator = gsp.generate_swap_MPO(accuhigh, acculow, mpo.L, [0, -1])
    accumulator = generate_id_MPO(accuhigh, acculow, mpo.L, 1)
    right_layer, left_right_inds, _ = right_layer_gen(
        accuhigh,
        mpo.upper_ind_id,
        mpo.L - 1,
        0,
        op_tag=Rop_tag,
        layer_tag=Rtag,
        left_right_index=True,
    )
    left_layer, _, left_left_inds = left_layer_gen(
        acculow,
        mpo.lower_ind_id,
        mpo.L - 1,
        0,
        op_tag=Lop_tag,
        layer_tag=Ltag,
        left_right_index=True,
    )
    (left_layer | accumulator | right_layer).draw()
    error = 1000
    output_map = {}
    for i in range(mpo.L + 1):
        output_map[accuhigh.format(i)] = acculow.format(i)
    Layer_ind_id = "LR{}"
    LRtags = [Layer_ind_id.format(i) for i in range(2 * mpo.L)]
    while error > precision and layercount < maxlayers:
        layer = left_layer | right_layer
        left_inds = {}
        for i, t in enumerate(layer):
            LRtagi = Layer_ind_id.format(i)
            if "R" in t.tags:
                for tag in t.tags:
                    try:
                        left_inds[LRtagi] = left_right_inds[tag]
                    except:
                        pass
            if "L" in t.tags:
                for tag in t.tags:
                    try:
                        left_inds[LRtagi] = left_left_inds[tag]
                    except:
                        pass
            t.add_tag(LRtagi)

        layer = layer_SVD_optimizer(
            layer, left_inds, accumulator, mpo, Layer_ind_id, 10, 1e-4
        )  # Current behavior is anomalous. we're using the same function, so it's the layer&accumulator combo that is at fault.
        layercount += 1
        if error > precision and layercount < maxlayers:
            next_accuhigh = uid.format(layercount) + "h{}"
            next_acculow = uid.format(layercount) + "l{}"
            rein_map = {}
            for i in range(mpo.L + 1):
                rein_map[mpo.upper_ind_id.format(i)] = next_accuhigh.format(i)
                rein_map[mpo.lower_ind_id.format(i)] = next_acculow.format(i)
            layer.reindex_(rein_map)
        else:
            next_accuhigh = mpo.upper_ind_id
            next_acculow = mpo.lower_ind_id

        left_layer = qtn.TensorNetwork(layer["L"])
        right_layer = qtn.TensorNetwork(layer["R"])
        mpo_left_layer = layer_gen.toMPO(
            left_layer, acculow, next_acculow, left_llid, Lop_tag
        )  # the indices are wrong
        mpo_right_layer = layer_gen.toMPO(
            right_layer, accuhigh, next_accuhigh, right_llid, Rop_tag
        )
        # print("mpo@L", mpo_layer.H@layer)
        circuit &= layer
        accuhigh = next_accuhigh
        acculow = next_acculow
        right_layer, left_right_inds, _ = right_layer_gen(
            accuhigh,
            mpo.upper_ind_id,
            mpo.L - 1,
            0,
            op_tag=Rop_tag,
            layer_tag=Rtag,
            left_right_index=True,
        )
        left_layer, _, left_left_inds = left_layer_gen(
            acculow,
            mpo.lower_ind_id,
            mpo.L - 1,
            0,
            op_tag=Lop_tag,
            layer_tag=Ltag,
            left_right_index=True,
        )

        accumulator.site_tag_id = "I{}"  # tag sanitization, because it seems .apply rely on this, but doesn't check.
        mpo_right_layer.site_tag_id = "I{}"
        mpo_left_layer.site_tag_id = "I{}"
        accumulator = mpo_right_layer.apply(accumulator.apply(mpo_left_layer))
        accumulator.upper_ind_id = "__TMP__{}"
        accumulator.lower_ind_id = acculow
        accumulator.upper_ind_id = accuhigh
        ####t'es rendu ici.
        accumulator = MPO_compressor(accumulator, precision * 0.01, precision * 0.02)
        # layer.draw()
        # accumulator.draw()
    # for i in range(mpo.L+1):
    #     output_map[acculow.format(i)] = mpo.lower_ind_id.format(i)
    #     output_map[accuhigh.format(i)] = mpo.upper_ind_id.format(i)

    accumulator.upper_ind_id = mpo.upper_ind_id
    accumulator.lower_ind_id = mpo.lower_ind_id

    circuit.reindex_(output_map)
    return circuit, accumulator


# Doing precisely what i describe in the article will require different definition of loss function and "MPO2Gates"

# Ne scale pas suffisament bien pour être très pratique.
# Il faut optimiser une couche à la fois, composer le résultat à haute précision avec le MPO cible (devrait réduire l'enchevetrement) et calculer la nouvelle couche avec le nouveau MPO ainsi créé.
def MPSO2Gates(
    mpo: qtn.MatrixProductOperator,
    precision,
    Nlayer,
    max_count=10,
    layer=generate_staircase_operators,
):
    nmpo = mpo.copy()
    sqrthalf = np.sqrt(0.5)
    for i, _ in enumerate(mpo):
        nmpo[i] = nmpo[i] * 0.5
    mpo = nmpo
    O = TwoqbitsLayers(mpo, Nlayer, layerGenerator=layer)
    L = generate_Lagrange_multipliers(O)
    loss = full_loss
    # print("O",O)
    # print("X",qX)
    # print("ts",ts)
    id_net = gen_id_net(mpo, sqrthalf)
    print("Initial error", Infidelity(O, mpo, id_net))
    id = qtn.Tensor(data=jnp.eye(4, 4).reshape(2, 2, 2, 2), inds=("a", "b", "c", "d"))
    optmzr = TNOptimizer(
        O,
        loss_fn=loss,
        # norm_fn=normalize_gates,
        loss_kwargs={"C": 0.0, "m": 50},
        loss_constants={
            "mpo": mpo,
            "id_net": id_net,
            "id": id,
            "L": L,
        },  # this is a constant TN to supply to loss_fn: psi,trivial_state, id, C,m)
        autodiff_backend="jax",  # {'jax', 'tensorflow', 'autograd','torch'}
        optimizer="L-BFGS-B",
        loss_target=precision,
    )
    error = 1000
    count = 0
    # print(gauge_regularizer(O,id,1/50,50))
    # print(Infidelity(O,qX,ts))
    OL_opt = optmzr.optimize(100)
    # OL_opt = normalize_gates(TN(OL_opt['O']))&TN(OL_opt['L'])
    while error > precision and count < max_count:
        optmzr.reset(OL_opt, loss_target=precision)
        val, grad = optmzr.vectorized_value_and_grad(optmzr.vectorizer.vector)
        print("initial gradient: ", np.mean(np.abs(grad)))
        # if (count%3==1):
        #     OL_opt = optmzr.optimize_basinhopping(n=200, nhop=10,temperature=0.5)
        # else:
        OL_opt = optmzr.optimize(20000, tol=precision)
        # OL_opt = normalize_gates(TN(OL_opt['O']))#&normalize_gates(TN(OL_opt['L']))
        error = Infidelity(OL_opt, mpo, id_net)
        # print("count: ", count)
        print(
            "current error: ",
            error,
            " unitarity error: ",
            unitarity_cost2(OL_opt, L, id),
        )
        count += 1
    O = normalize_gates(OL_opt)
    return O, error


# %%
