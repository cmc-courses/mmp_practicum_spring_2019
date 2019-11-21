import gf
import numpy as np


class BCH:
    def __init__(self, n, t):
        self.n, self.t, self.q = n, t, int(np.log2(n + 1))

        with open('./primpoly.txt', 'r') as file:
            primpolys = np.array(list(map(int, file.readline().strip().split(','))))
            primpolys = primpolys[np.argwhere(np.log2(primpolys).astype(np.int) == self.q)].reshape(-1)

        self.pm = gf.gen_pow_matrix(primpolys[0])
        self.R = self.pm[0:2 * t, 1]
        self.g, _ = gf.minpoly(self.R, self.pm)

        self.m, self.k = self.g.shape[0] - 1, self.n - (self.g.shape[0] - 1)

    def _encode(self, u):
        assert self.n - (self.g.shape[0] - 1) == u.shape[0]
        x_power = np.zeros_like(self.g, dtype=np.int)
        x_power[0] = 1
        x_power_u = gf.polyprod(x_power, u, self.pm)
        _, mod = gf.polydivmod(x_power_u, self.g, self.pm)
        result = gf.polyadd(x_power_u, mod)
        result = np.concatenate([np.zeros([self.n - result.shape[0]], dtype=np.int), result])
        return result

    def encode(self, u):
        return np.array(list(map(self._encode, u)))

    def _decode(self, w, method):
        t = self.R.shape[0] // 2
        syndromes = gf.polyval(w, self.R, self.pm)
        if np.sum(syndromes != 0) == 0:
            return w

        if method == 'pgz':
            lambda_ = np.nan
            for nu in range(t, 0, -1):
                a = np.array([[syndromes[j] for j in range(i, nu + i)] for i in range(nu)], dtype=np.int)
                b = np.array([syndromes[i] for i in range(nu, 2 * nu)], dtype=np.int)
                lambda_ = gf.linsolve(a, b, self.pm)
                if lambda_ is not np.nan:
                    break
            if lambda_ is np.nan:
                return np.full(self.n, np.nan, dtype=np.int)
            lambda_ = np.concatenate([lambda_, [1]])
        elif method == 'euclid':
            z = np.zeros([2 * t + 2], dtype=np.int)
            z[0] = 1
            syndromic_polynom = np.concatenate([syndromes[::-1], [1]])
            _, _, lambda_ = gf.euclid(z, syndromic_polynom, self.pm, max_deg=t)
        else:
            raise ValueError

        n_roots = 0
        locators_values = gf.polyval(lambda_, np.arange(1, self.n + 1), self.pm)
        for idx in range(1, self.n + 1):
            if not locators_values[idx - 1]:
                position = self.n - self.pm[gf.divide(1, idx, self.pm) - 1, 0] - 1
                w[position] = 1 - w[position]
                n_roots += 1
        if n_roots != lambda_.shape[0] - 1:
            return np.full(self.n, np.nan, dtype=np.int)
        return w

    def decode(self, w, method='euclid'):
        return np.array(list(map(lambda x: self._decode(x, method), w)))

    def dist(self):
        k = self.n - (self.g.shape[0] - 1)
        result = np.inf
        for value in range(1, 2 ** k):
            block = np.array(list(map(int, bin(value)[2:])))
            block = np.concatenate([np.zeros([k - block.shape[0]], dtype=np.int), block])
            result = min(np.count_nonzero(self._encode(block)), result)
        return int(result)
