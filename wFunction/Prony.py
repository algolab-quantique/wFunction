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

import numpy as np


"""
This approach to interpolation doesn't make for a very efficient high precision reprentation of a function such as the Gaussian. It's
"""


def maximal_size_linear_predictor(func: np.array):
    d = np.zeros(len(func) - 1)
    d[0] = func[1] / func[0]
    assert False, "to be finished later."


def Linear_predictor(func: np.array, m: int):
    """Fait l'analyse de Prony de la fonction donné en argument avec au plus m termes. Suppose un pas temporelle uniforme égale à 1."""
    assert len(func) > m
    assert m > 0
    Y = np.array(
        [
            [func[n - j] if n >= j else 0 for n, y in enumerate(func[:-1])]
            for j in range(m)
        ]
    ).transpose()
    return np.linalg.lstsq(Y, func[1:], rcond=None)[:-2]


def predict(func: np.array, d: np.array):
    l = min(len(d), len(func))
    return np.sum(d[:l] * func[-1 : -l - 1 : -1])


def predict_N(func: np.array, d: np.array, N: int):
    l = len(d)
    f = len(func)
    out = np.concatenate([func, np.zeros(N)])
    for j in range(f, f + N):
        out[j] = predict(out[:j], d)
    return out


def extract_freqs(d):
    coeff = np.concatenate([-d[-1::-1], [1]])
    return np.roots(coeff)


def Prony_amplitude(freq: np.array, func: np.array):
    t = np.expand_dims(np.array([i for i, f in enumerate(func)]), 1)
    F = np.power(freq, t)
    return np.linalg.lstsq(F, func, rcond=None)[:-2]


def Prony_analysis(func: np.array, m: int):
    d, freq_error = Linear_predictor(func, m)
    freq = 1 / extract_freqs(d).conj()
    A, A_error = Prony_amplitude(freq, func)
    return freq, A, freq_error, A_error


def Prony_reconstruct(t: np.array, A: np.array, F: np.array):
    if len(t.shape) == 1:
        tt = np.expand_dims(t, 1)
    else:
        tt = t
    return np.dot(np.power(F, tt), A)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    M = 600
    sampling = np.linspace(-2, 2, M)
    w = np.linspace(-2, 2, 2000)
    w2 = np.linspace(0, M - 1, 2000)

    samples = np.exp(-(sampling**2) / 2) / np.sqrt(2 * np.pi)
    f = np.exp(-(w**2) / 2) / np.sqrt(2 * np.pi)
    freq, Amplitude, ferr, Aerr = Prony_analysis(samples, M - 4)
    print("frequency error: ", ferr, "amplitude error: ", Aerr)
    samples_bis = np.dot(
        np.array([freq**i for i, f in enumerate(samples)]), Amplitude
    )

    plt.plot(np.imag(np.log(freq)), np.real(np.log(freq)), "bo", label="cplx freqs")
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(sampling, samples, label="samples")
    plt.plot(sampling, samples_bis, label="bis")
    plt.legend()
    plt.show()

    f_bis = Prony_reconstruct(w2, Amplitude, freq)
    plt.plot(w, f, label="original")
    plt.plot(w, f_bis, label="reproduction")
    plt.legend()
    plt.show()

    plt.semilogy(w, np.abs(f - f_bis))
    plt.show()
