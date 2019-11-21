import numpy as np


def gen_pow_matrix(primpoly):
    poly_power = len(bin(primpoly)) - 3
    result = np.empty([2 ** poly_power - 1, 2], dtype=np.int)

    current_poly = 0b10
    for i in range(result.shape[0]):
        result[current_poly - 1, 0] = i + 1
        result[i, 1] = current_poly

        current_poly <<= 1
        if current_poly >> poly_power:
            current_poly ^= primpoly

    return result


def add(x, y):
    return np.bitwise_xor(x, y)


def sum(x, axis=0):
    return np.bitwise_xor.reduce(x, axis=axis)


def prod(x, y, pm):
    x_pow, y_pow = pm[:, 0][x - 1], pm[:, 0][y - 1]
    result = pm[:, 1][np.mod(x_pow + y_pow, pm.shape[0]) - 1]
    if isinstance(result, np.integer) or result.size == 1:
        if np.logical_or(x == 0, y == 0):
            result = np.array(0)
    else:
        result[np.logical_or(x == 0, y == 0)] = 0
    return result


def divide(x, y, pm):
    x_pow, y_pow = pm[:, 0][x - 1], pm[:, 0][y - 1]
    result = pm[:, 1][np.mod(x_pow - y_pow, pm.shape[0]) - 1]
    if isinstance(result, np.integer) or x.size == 1:
        if x == 0:
            result *= 0
    else:
        result[x == 0] = 0
    assert np.any(y != 0)
    return result


def linsolve(a, b, pm):
    a, b = np.copy(a), np.copy(b)
    result = np.empty_like(b)
    for idx in range(a.shape[0]):
        nonzero_idx = np.nonzero(a[idx:, idx])[0]
        if len(nonzero_idx) == 0:
            return np.nan
        nonzero_idx = nonzero_idx[0] + idx
        a[:, idx:][[idx, nonzero_idx]] = a[:, idx:][[nonzero_idx, idx]]
        b[idx], b[nonzero_idx] = b[nonzero_idx], b[idx]
        for jdx in range(idx + 1, a.shape[0]):
            coefficient = divide(a[jdx, idx], a[idx, idx], pm)
            a[jdx, idx:], b[jdx] = (
                add(a[jdx, idx:], prod(a[idx, idx:], coefficient, pm)),
                add(b[jdx],  prod(b[idx], coefficient, pm))
            )
    result[-1] = divide(b[-1], a[-1, -1], pm)
    for idx in range(a.shape[0] - 2, -1, -1):
        result[idx] = divide(add(b[idx], sum(prod(a[idx, idx + 1:], result[idx + 1:], pm))), a[idx, idx], pm)
    return result


def polynorm(p):
    if np.sum(p != 0) == 0:
        return np.array([0])
    return p[np.nonzero(p)[0][0]:]


def minpoly(x, pm):
    roots = set(x)
    for root in x:
        next_root = prod(root, root, pm)
        while next_root != root:
            roots.add(next_root)
            next_root = prod(next_root, next_root, pm)

    result = np.array([1])
    for root in roots:
        result = polyprod(result, np.array([1, root]), pm)
    return result, np.array(list(roots))


def polyval(p, x, pm):
    x_powers = np.ones([x.shape[0], p.shape[0]], dtype=np.int)
    for idx in range(1, x_powers.shape[1]):
        x_powers[:, idx] = prod(x_powers[:, idx - 1], x, pm)
    return sum(prod(p[::-1][np.newaxis, :], x_powers, pm), axis=1)


def polyadd(p1, p2, reduce=True):
    if p1.shape[0] > p2.shape[0]:
        p1, p2 = p2, p1
    p1 = np.concatenate([np.zeros(p2.shape[0] - p1.shape[0], dtype=np.int), p1])
    result = add(p1, p2)
    if reduce:
        return polynorm(result)
    return result


def polyprod(p1, p2, pm):
    p1, p2 = polynorm(p1), polynorm(p2)

    result = np.zeros([p1.shape[0] + p2.shape[0] - 1], dtype=np.int)
    for idx1, value1 in enumerate(p1):
        for idx2, value2 in enumerate(p2):
            result[idx1 + idx2] = add(result[idx1 + idx2], prod(value1, value2, pm))
    return polynorm(result)


def polydivmod(p1, p2, pm):
    p1, p2 = polynorm(p1), polynorm(p2)
    if p2[0] == 0:
        raise BaseException

    if p1.shape[0] - p2.shape[0] < 0:
        return np.array([0]), p1

    result = np.zeros(p1.shape[0] - p2.shape[0] + 1, dtype=np.int)
    for idx in range(result.shape[0]):
        if not p1[idx]:
            continue
        if np.sum(p1 != 0) == 0:
            break
        result[idx] = divide(p1[idx], p2[0], pm)
        multiplier = np.concatenate([
            polyprod(result[idx:idx + 1], p2, pm),
            np.zeros([p1.shape[0] - p2.shape[0] - idx], dtype=np.int)
        ])
        p1 = polyadd(p1, multiplier, reduce=False)
    return result, polynorm(p1)


def euclid(p1, p2, pm, max_deg=0):
    coefficients = [
        [np.array([1]), np.array([0])],
        [np.array([0]), np.array([1])]
    ]
    while p2.shape[0] - 1 > max_deg:
        q, r = polydivmod(p1, p2, pm)
        coefficients = [
            [coefficients[0][1], polyadd(coefficients[0][0], polyprod(q, coefficients[0][1], pm))],
            [coefficients[1][1], polyadd(coefficients[1][0], polyprod(q, coefficients[1][1], pm))]
        ]
        p1, p2 = p2, r
    return p2, coefficients[0][1], coefficients[1][1]
