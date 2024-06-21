# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import numpy as np
import openblas_wrap as ow


# ### BLAS level 1 ###

# dnrm2

dnrm2_sizes = [100, 200, 400, 600, 800, 1000]

def run_dnrm2(n, x, incx, func):
    res = func(x, n, incx=incx)
    return res



class Nrm2:

    params = [dnrm2_sizes, ['d', 'dz']]
    param_names = ["size", "variant"]

    def setup(self, n, variant):
        rndm = np.random.RandomState(1234)
        self.x = rndm.uniform(size=(n,)).astype(float)
        self.nrm2 = ow.get_func('nrm2', variant)

    def time_dnrm2(self, n, variant):
        run_dnrm2(n, self.x, 1, self.nrm2)


# ddot

ddot_sizes = [100, 200, 400, 600, 800, 1000]

def run_ddot(x, y, func):
    res = func(x, y)
    return res


class DDot:
    params = ddot_sizes
    param_names = ["size"]

    def setup(self, n):
        rndm = np.random.RandomState(1234)
        self.x = np.array(rndm.uniform(size=(n,)), dtype=float)
        self.y = np.array(rndm.uniform(size=(n,)), dtype=float)
        self.func = ow.get_func('dot', 'd')

    def time_ddot(self, n):
        run_ddot(self.x, self.y, self.func)



# daxpy

daxpy_sizes = [100, 200, 400, 600, 800, 1000]

def run_daxpy(x, y, func):
    res = func(x, y, a=2.0)
    return res


class Daxpy:
    params = [daxpy_sizes, ['s', 'd', 'c', 'z']]
    param_names = ["size", "variant"]

    def setup(self, n, variant):
        rndm = np.random.RandomState(1234)
        self.x = np.array(rndm.uniform(size=(n,)), dtype=float)
        self.y = np.array(rndm.uniform(size=(n,)), dtype=float)
        self.axpy = ow.get_func('axpy', variant)

    def time_daxpy(self, n, variant):
        run_daxpy(self.x, self.y, self.axpy)



# ### BLAS level 3 ###

# dgemm

gemm_sizes = [100, 200, 400, 600, 800, 1000]

def run_dgemm(a, b, c, func):
    alpha = 1.0
    res = func(alpha, a, b, c=c, overwrite_c=True)
    return res


class Dgemm:
    params = [gemm_sizes, ['s', 'd', 'c', 'z']]
    param_names = ["size", 'variant']

    def setup(self, n, variant):
        rndm = np.random.RandomState(1234)
        self.a = np.array(rndm.uniform(size=(n, n)), dtype=float, order='F')
        self.b = np.array(rndm.uniform(size=(n, n)), dtype=float, order='F')
        self.c = np.empty((n, n), dtype=float, order='F')
        self.func = ow.get_func('gemm', variant)

    def time_dgemm(self, n, variant):
        run_dgemm(self.a, self.b, self.c, self.func)


# dsyrk

syrk_sizes = [100, 200, 400, 600, 800, 1000]


def run_dsyrk(a, c, func):
    res = func(1.0, a, c=c, overwrite_c=True)
    return res


class DSyrk:
    params = [syrk_sizes, ['s', 'd', 'c', 'z']]
    param_names = ["size", "variant"]

    def setup(self, n, variant):
        rndm = np.random.RandomState(1234)
        self.a = np.array(rndm.uniform(size=(n, n)), dtype=float, order='F')
        self.c = np.empty((n, n), dtype=float, order='F')
        self.func = ow.get_func('syrk', variant)

    def time_dsyrk(self, n, variant):
        run_dsyrk(self.a, self.c, self.func)


# ### LAPACK ###

# linalg.solve

dgesv_sizes = [100, 200, 400, 600, 800, 1000]


def run_dgesv(a, b, func):
    res = func(a, b, overwrite_a=True, overwrite_b=True)
    return res


class Dgesv:
    params = [dgesv_sizes, ['s', 'd', 'c', 'z']]
    param_names = ["size", "variant"]

    def setup(self, n, variant):
        rndm = np.random.RandomState(1234)
        self.a = (np.array(rndm.uniform(size=(n, n)), dtype=float, order='F') +
                  np.eye(n, order='F'))
        self.b = np.array(rndm.uniform(size=(n, 1)), order='F')
        self.func = ow.get_func('gesv', variant)

    def time_dgesv(self, n, variant):
        run_dgesv(self.a, self.b, self.func)

      # XXX: how to run asserts?
      #  lu, piv, x, info = benchmark(run_gesv, a, b)
      #  assert lu is a
      #  assert x is b
      #  assert info == 0


# linalg.svd

dgesdd_sizes = ["100, 5", "1000, 222"]


def run_dgesdd(a, lwork, func):
    res = func(a, lwork=lwork, full_matrices=False, overwrite_a=False)
    return res


class Dgesdd:
    params = [dgesdd_sizes, ['s', 'd']]
    param_names = ["(m, n)", "variant"]

    def setup(self, mn, variant):
        m, n = (int(x) for x in mn.split(","))

        rndm = np.random.RandomState(1234)
        a = np.array(rndm.uniform(size=(m, n)), dtype=float, order='F')

        gesdd_lwork = ow.get_func('gesdd_lwork', variant)

        lwork, info = gesdd_lwork(m, n)
        lwork = int(lwork)
        assert info == 0

        self.a, self.lwork = a, lwork
        self.func = ow.get_func('gesdd', variant)

    def time_dgesdd(self, mn, variant):
        run_dgesdd(self.a, self.lwork, self.func)


# linalg.eigh

dsyev_sizes = [50, 64, 128, 200]


def run_dsyev(a, lwork, func):
    res = func(a, lwork=lwork, overwrite_a=True)
    return res


class Dsyev:
    params = [dsyev_sizes, ['s', 'd']]
    param_names = ["size", "variant"]

    def setup(self, n, variant):
        rndm = np.random.RandomState(1234)
        a = rndm.uniform(size=(n, n))
        a = np.asarray(a + a.T, dtype=float, order='F')
        a_ = a.copy()

        syev_lwork = ow.get_func('syev_lwork', variant)
        lwork, info = syev_lwork(n)
        lwork = int(lwork)
        assert info == 0

        self.a = a_
        self.lwork = lwork
        self.func = ow.get_func('syev', variant)

    def time_dsyev(self, n, variant):
        run_dsyev(self.a, self.lwork, self.func)

