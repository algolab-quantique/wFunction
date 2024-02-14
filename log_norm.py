#%%
import wFunction as wf
import numpy as np
import qiskit as qs
from qiskit.circuit import qpy_serialization

#%%
from scipy.stats import lognorm as scplog


def lognorm(x, mu, sigma):
    return scplog.pdf(np.exp(-mu) * x, sigma)


#%%
nqbit = 5
threeqb = qs.QuantumRegister(nqbit)
circ = wf.Generate_f_circuit(
    lambda x: (lognorm(x, 1, 1)),
    1e-5,
    1e-12,
    nqbit,
    [0.00000001, 7],
    threeqb,
    1,
    "lognormal",
)
circ.draw("mpl")
#%%
filename = "lognorm_{}q.qpy".format(nqbit)
with open(filename, "wb") as fd:
    qpy_serialization.dump(circ, fd)
# %%

from qiskit import transpile
from qiskit.providers.aer import QasmSimulator

simulator = QasmSimulator()
#%%
tr_circ = transpile(circ, simulator)

tr_circ.reverse_bits()
tr_circ.draw("mpl")
#%%
tr_circ = transpile(tr_circ.decompose(), simulator, ["cx", "id", "rz", "sx", "x"])

#%%
tr_circ.decompose().draw("mpl")
#%%
tr_circ.qasm(True, "3qblognorm11.qasm")

# %%

import numpy as np

from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_finance.circuit.library import LogNormalDistribution


# number of qubits to represent the uncertainty
num_uncertainty_qubits = 3

# parameters for considered random distribution
S = 2.0  # initial spot price
vol = 0.4  # volatility of 40%
r = 0.05  # annual interest rate of 4%
T = 40 / 365  # 40 days to maturity

# resulting parameters for log-normal distribution
mu = (r - 0.5 * vol**2) * T + np.log(S)
sigma = vol * np.sqrt(T)
mean = np.exp(mu + sigma**2 / 2)
variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
stddev = np.sqrt(variance)

# lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
low = np.maximum(0, mean - 3 * stddev)
high = mean + 3 * stddev

# construct circuit for uncertainty model
uncertainty_model = LogNormalDistribution(
    num_uncertainty_qubits, mu=mu, sigma=sigma**2, bounds=(low, high)
)
uncertainty_model.draw("mpl")
#%%

# set the strike price (should be within the low and the high value of the uncertainty)
strike_price_1 = 1.438
strike_price_2 = 2.584

# set the approximation scaling for the payoff function
rescaling_factor = 0.25

# setup piecewise linear objective fcuntion
breakpoints = [low, strike_price_1, strike_price_2]
slopes = [0, 1, 0]
offsets = [0, 0, strike_price_2 - strike_price_1]
f_min = 0
f_max = strike_price_2 - strike_price_1
bull_spread_objective = LinearAmplitudeFunction(
    num_uncertainty_qubits,
    slopes,
    offsets,
    domain=(low, high),
    image=(f_min, f_max),
    breakpoints=breakpoints,
    rescaling_factor=rescaling_factor,
)
bull_spread_objective.draw("mpl")
#%%
# construct A operator for QAE for the payoff function by
# composing the uncertainty model and the objective
bull_spread = bull_spread_objective.compose(uncertainty_model, front=True)


bull_spread.draw("mpl")
# %%
cr = qs.ClassicalRegister(1)
bull_spread.add_register(cr)
bull_spread.measure(3, 0)
# %%
bull_spread.draw("mpl")
# %%
