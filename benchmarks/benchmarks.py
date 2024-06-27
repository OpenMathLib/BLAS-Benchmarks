# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import numpy as np
import openblas_wrap as ow

dtype_map = {
    's': np.float32,
    'd': np.float64,
    'c': np.complex64,
    'z': np.complex128,
    'dz': np.complex128,
}


# ### BLAS level 1 ###

# nrm2

nrm2_sizes = [100, 200, 400, 600, 800, 1000]

def run_nrm2(n, x, incx, func):
    res = func(x, n, incx=incx)
    return res



class Nrm2:

    params = [nrm2_sizes, ['d', 'dz']]
    param_names = ["size", "variant"]

    def setup(self, n, variant):
        rndm = np.random.RandomState(1234)
        dtyp = dtype_map[variant]

        self.x = rndm.uniform(size=(n,)).astype(dtyp)
        self.nrm2 = ow.get_func('nrm2', variant)

    def time_nrm2(self, n, variant):
        run_nrm2(n, self.x, 1, self.nrm2)


# dot

dot_sizes = [100, 200, 400, 600, 800, 1000]

def run_dot(x, y, func):
    res = func(x, y)
    return res


class dot:
    params = dot_sizes
    param_names = ["size"]

    def setup(self, n):
        rndm = np.random.RandomState(1234)
        dtyp = float

        self.x = np.array(rndm.uniform(size=(n,)), dtype=dtyp)
        self.y = np.array(rndm.uniform(size=(n,)), dtype=dtyp)
        self.func = ow.get_func('dot', 'd')

    def time_dot(self, n):
        run_dot(self.x, self.y, self.func)



# axpy

axpy_sizes = [100, 200, 400, 600, 800, 1000]

def run_axpy(x, y, func):
    res = func(x, y, a=2.0)
    return res


class axpy:
    params = [axpy_sizes, ['s', 'd', 'c', 'z']]
    param_names = ["size", "variant"]

    def setup(self, n, variant):
        rndm = np.random.RandomState(1234)
        dtyp = dtype_map[variant]

        self.x = np.array(rndm.uniform(size=(n,)), dtype=dtyp)
        self.y = np.array(rndm.uniform(size=(n,)), dtype=dtyp)
        self.axpy = ow.get_func('axpy', variant)

    def time_axpy(self, n, variant):
        run_axpy(self.x, self.y, self.axpy)



# ### BLAS level 2 ###

# gemv

gemv_sizes = [100, 200, 400, 600, 800, 1000]

def run_gemv(a, x, y, func):
    res = func(1.0, a, x, y=y, overwrite_y=True)
    return res


class gemv:
    params = [gemv_sizes, ['s', 'd', 'c', 'z']]
    param_names = ["size", "variant"]

    def setup(self, n, variant):
        rndm = np.random.RandomState(1234)
        dtyp = dtype_map[variant]

        self.a = np.array(rndm.uniform(size=(n,n)), dtype=dtyp)
        self.x = np.array(rndm.uniform(size=(n,)), dtype=dtyp)
        self.y = np.zeros(n, dtype=dtyp)

        self.gemv = ow.get_func('gemv', variant)

    def time_gemv(self, n, variant):
        run_gemv(self.a, self.x, self.y, self.gemv)


# gbmv

gbmv_sizes = [100, 200, 400, 600, 800, 1000]

def run_gbmv(m, n, kl, ku, a, x, y, func):
    res = func(m, n, kl, ku, 1.0, a, x, y=y, overwrite_y=True)
    return res


class gbmv:
    params = [gbmv_sizes, ['s', 'd', 'c', 'z'], [1, 2, 3]]
    param_names = ["size", "variant", "kl"]

    def setup(self, n, variant, kl):
        rndm = np.random.RandomState(1234)
        dtyp = dtype_map[variant]

        self.x = np.array(rndm.uniform(size=(n,)), dtype=dtyp)
        self.y = np.empty(n, dtype=dtyp)

        self.m = n

        a = rndm.uniform(size=(2*kl + 1, n))
        self.a = np.array(a, dtype=dtyp, order='F')

        self.gbmv = ow.get_func('gbmv', variant)

    def time_gbmv(self, n, variant, kl):
        run_gbmv(self.m, n, kl, kl, self.a, self.x, self.y, self.gbmv)


# ### BLAS level 3 ###

# gemm

gemm_sizes = [100, 200, 400, 600, 800, 1000]

def run_gemm(a, b, c, func):
    alpha = 1.0
    res = func(alpha, a, b, c=c, overwrite_c=True)
    return res


class gemm:
    params = [gemm_sizes, ['s', 'd', 'c', 'z']]
    param_names = ["size", 'variant']

    def setup(self, n, variant):
        rndm = np.random.RandomState(1234)
        dtyp = dtype_map[variant]

        self.a = np.array(rndm.uniform(size=(n, n)), dtype=dtyp, order='F')
        self.b = np.array(rndm.uniform(size=(n, n)), dtype=dtyp, order='F')
        self.c = np.empty((n, n), dtype=dtyp, order='F')
        self.func = ow.get_func('gemm', variant)

    def time_gemm(self, n, variant):
        run_gemm(self.a, self.b, self.c, self.func)


# syrk

syrk_sizes = [100, 200, 400, 600, 800, 1000]


def run_syrk(a, c, func):
    res = func(1.0, a, c=c, overwrite_c=True)
    return res


class syrk:
    params = [syrk_sizes, ['s', 'd', 'c', 'z']]
    param_names = ["size", "variant"]

    def setup(self, n, variant):
        rndm = np.random.RandomState(1234)
        dtyp = dtype_map[variant]

        self.a = np.array(rndm.uniform(size=(n, n)), dtype=dtyp, order='F')
        self.c = np.empty((n, n), dtype=dtyp, order='F')
        self.func = ow.get_func('syrk', variant)

    def time_syrk(self, n, variant):
        run_syrk(self.a, self.c, self.func)


# ### LAPACK ###

# linalg.solve

gesv_sizes = [100, 200, 400, 600, 800, 1000]


def run_gesv(a, b, func):
    res = func(a, b, overwrite_a=True, overwrite_b=True)
    return res


class gesv:
    params = [gesv_sizes, ['s', 'd', 'c', 'z']]
    param_names = ["size", "variant"]

    def setup(self, n, variant):
        rndm = np.random.RandomState(1234)
        dtyp = dtype_map[variant]

        self.a = (np.array(rndm.uniform(size=(n, n)), dtype=dtyp, order='F') +
                  np.eye(n, dtype=dtyp, order='F'))
        self.b = np.array(rndm.uniform(size=(n, 1)), dtype=dtyp, order='F')
        self.func = ow.get_func('gesv', variant)

    def time_gesv(self, n, variant):
        run_gesv(self.a, self.b, self.func)

      # XXX: how to run asserts?
      #  lu, piv, x, info = benchmark(run_gesv, a, b)
      #  assert lu is a
      #  assert x is b
      #  assert info == 0


# linalg.svd

gesdd_sizes = ["100, 5", "1000, 222"]


def run_gesdd(a, lwork, func):
    res = func(a, lwork=lwork, full_matrices=False, overwrite_a=False)
    return res


class gesdd:
    params = [gesdd_sizes, ['s', 'd']]
    param_names = ["(m, n)", "variant"]

    def setup(self, mn, variant):
        m, n = (int(x) for x in mn.split(","))

        rndm = np.random.RandomState(1234)
        dtyp = dtype_map[variant]

        a = np.array(rndm.uniform(size=(m, n)), dtype=dtyp, order='F')

        gesdd_lwork = ow.get_func('gesdd_lwork', variant)

        lwork, info = gesdd_lwork(m, n)
        lwork = int(lwork)
        assert info == 0

        self.a, self.lwork = a, lwork
        self.func = ow.get_func('gesdd', variant)

    def time_gesdd(self, mn, variant):
        run_gesdd(self.a, self.lwork, self.func)


# linalg.eigh

syev_sizes = [50, 64, 128, 200]


def run_syev(a, lwork, func):
    res = func(a, lwork=lwork, overwrite_a=True)
    return res


class syev:
    params = [syev_sizes, ['s', 'd']]
    param_names = ["size", "variant"]

    def setup(self, n, variant):
        rndm = np.random.RandomState(1234)
        dtyp = dtype_map[variant]

        a = rndm.uniform(size=(n, n))
        a = np.asarray(a + a.T, dtype=dtyp, order='F')
        a_ = a.copy()

        syev_lwork = ow.get_func('syev_lwork', variant)
        lwork, info = syev_lwork(n)
        lwork = int(lwork)
        assert info == 0

        self.a = a_
        self.lwork = lwork
        self.func = ow.get_func('syev', variant)

    def time_syev(self, n, variant):
        run_syev(self.a, self.lwork, self.func)

