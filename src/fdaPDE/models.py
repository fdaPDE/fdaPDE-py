## This file is part of fdaPDE, a C++ library for physics-informed
## spatial and functional data analysis.
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

from fdaPDE import cpp as _cpp
from fdaPDE.fem import FeFunction
from .formula import _formula
import numpy as np

__all__ = ["SRPDE", "GSRPDE", "QSRPDE", "gcv", "grid_search", "fe_elliptic"]


def gcv(optimizer, edf="stochastic", mc_samples=100, seed=None):
    args = {"mc_samples": mc_samples, "seed": -1 if seed == None else seed}
    args.update(optimizer)
    return args


def grid_search(grid):
    import numpy as np

    args = {"opt": "grid", "grid": np.asarray(grid)}
    return args


def fe_elliptic(K=None, b=None, c=None, u=None):
    return {"K": K, "b": b, "c": c, "u": u}


class SRPDE:
    def __init__(self, formula, data, penalty):
        import __main__

        # parse formula and detect spatial field object
        tmp = _formula(formula)
        f_candidates = [v for v in tmp.rhs if v not in data.colnames]
        if len(f_candidates) != 1:
            raise ValueError("Expected exactly one nonparametric term.")
        f_symbol = f_candidates[0]
        f_obj = getattr(__main__, f_symbol, None)
        if f_obj is not None:
            if not isinstance(f_obj, FeFunction):
                raise TypeError(f"Global symbol '{f_symbol}' is not a spatial field.")
        else:
            raise NameError(f"Global symbol '{f_symbol}' not found.")
        self._f = f_obj  # store reference to field

        if penalty == None:
            self._ptr = _cpp.models.SRPDE(formula, data._ptr, None)
        else:
            mesh = data.mesh
            embed_dim = mesh.dim[1]
            quad_nodes = _cpp.fem.fe_simplex_2d_p1_quadrature(mesh)
            n_quad_nodes = quad_nodes.shape[0]

            # expand PDE parameters
            params = {}
            fields = {
                "K": embed_dim * embed_dim,
                "b": embed_dim,
                "c": 1,
                "u": 1,
            }
            for name, size in fields.items():
                value = penalty.get(name)
                if value is None:
                    params[name] = (
                        np.zeros((n_quad_nodes, size))
                        if size > 1
                        else np.zeros((n_quad_nodes, size)).reshape(-1)
                    )
                else:
                    if callable(value):
                        params[name] = (
                            np.asarray(value(np.asarray(quad_nodes)))
                            if size > 1
                            else np.asarray(value(np.asarray(quad_nodes))).reshape(-1)
                        )
                    else:
                        params[name] = (
                            np.tile(np.asarray(value).flatten(), (n_quad_nodes, 1))
                            if size > 1
                            else np.tile(
                                np.asarray(value).flatten(), (n_quad_nodes, 1)
                            ).reshape(-1)
                        )

            self._ptr = _cpp.models.SRPDE(formula, data._ptr, params)

    def fit(self, lambda_=None, calibration_=None):
        r = None
        if calibration_ == None:
            self._ptr.fit(lambda_)
        else:
            r = self._ptr.fit_gcv(calibration_)
        self._f.set_coeff(self._ptr.f())
        if r != None:
            return r

    @property
    def f(self):
        return self._ptr.f()

    @property
    def beta(self):
        return self._ptr.beta()

    @property
    def fitted(self):
        return self._ptr.fitted()


class GSRPDE:
    def __init__(self, formula, data, family, penalty):
        import __main__

        # parse formula and detect spatial field object
        tmp = _formula(formula)
        f_candidates = [v for v in tmp.rhs if v not in data.colnames]
        if len(f_candidates) != 1:
            raise ValueError("Expected exactly one nonparametric term.")
        f_symbol = f_candidates[0]
        f_obj = getattr(__main__, f_symbol, None)
        if f_obj is not None:
            if not isinstance(f_obj, FeFunction):
                raise TypeError(f"Global symbol '{f_symbol}' is not a spatial field.")
        else:
            raise NameError(f"Global symbol '{f_symbol}' not found.")
        self._f = f_obj  # store reference to field

        if penalty == None:
            self._ptr = _cpp.models.GSRPDE(formula, data._ptr, family, None)
        else:
            mesh = data.mesh
            embed_dim = mesh.dim[1]
            quad_nodes = _cpp.fem.fe_simplex_2d_p1_quadrature(mesh)
            n_quad_nodes = quad_nodes.shape[0]

            # expand PDE parameters
            params = {}
            fields = {
                "K": embed_dim * embed_dim,
                "b": embed_dim,
                "c": 1,
                "u": 1,
            }
            for name, size in fields.items():
                value = penalty.get(name)
                if value is None:
                    params[name] = (
                        np.zeros((n_quad_nodes, size))
                        if size > 1
                        else np.zeros((n_quad_nodes, size)).reshape(-1)
                    )
                else:
                    if callable(value):
                        params[name] = (
                            np.asarray(value(np.asarray(quad_nodes)))
                            if size > 1
                            else np.asarray(value(np.asarray(quad_nodes))).reshape(-1)
                        )
                    else:
                        params[name] = (
                            np.tile(np.asarray(value).flatten(), (n_quad_nodes, 1))
                            if size > 1
                            else np.tile(
                                np.asarray(value).flatten(), (n_quad_nodes, 1)
                            ).reshape(-1)
                        )

            self._ptr = _cpp.models.GSRPDE(formula, data._ptr, family, params)

    def fit(self, lambda_=None, calibration_=None):
        r = None
        if calibration_ == None:
            self._ptr.fit(lambda_)
        else:
            r = self._ptr.fit_gcv(calibration_)
        self._f.set_coeff(self._ptr.f())
        if r != None:
            return r

    @property
    def f(self):
        return self._ptr.f()

    @property
    def beta(self):
        return self._ptr.beta()

    @property
    def fitted(self):
        return self._ptr.fitted()

    
class QSRPDE:
    def __init__(self, formula, data, level, penalty):
        import __main__

        # parse formula and detect spatial field object
        tmp = _formula(formula)
        f_candidates = [v for v in tmp.rhs if v not in data.colnames]
        if len(f_candidates) != 1:
            raise ValueError("Expected exactly one nonparametric term.")
        f_symbol = f_candidates[0]
        f_obj = getattr(__main__, f_symbol, None)
        if f_obj is not None:
            if not isinstance(f_obj, FeFunction):
                raise TypeError(f"Global symbol '{f_symbol}' is not a spatial field.")
        else:
            raise NameError(f"Global symbol '{f_symbol}' not found.")
        self._f = f_obj  # store reference to field

        if penalty == None:
            self._ptr = _cpp.models.QSRPDE(formula, data._ptr, level, None)
        else:
            mesh = data.mesh
            embed_dim = mesh.dim[1]
            quad_nodes = _cpp.fem.fe_simplex_2d_p1_quadrature(mesh)
            n_quad_nodes = quad_nodes.shape[0]

            # expand PDE parameters
            params = {}
            fields = {
                "K": embed_dim * embed_dim,
                "b": embed_dim,
                "c": 1,
                "u": 1,
            }
            for name, size in fields.items():
                value = penalty.get(name)
                if value is None:
                    params[name] = (
                        np.zeros((n_quad_nodes, size))
                        if size > 1
                        else np.zeros((n_quad_nodes, size)).reshape(-1)
                    )
                else:
                    if callable(value):
                        params[name] = (
                            np.asarray(value(np.asarray(quad_nodes)))
                            if size > 1
                            else np.asarray(value(np.asarray(quad_nodes))).reshape(-1)
                        )
                    else:
                        params[name] = (
                            np.tile(np.asarray(value).flatten(), (n_quad_nodes, 1))
                            if size > 1
                            else np.tile(
                                np.asarray(value).flatten(), (n_quad_nodes, 1)
                            ).reshape(-1)
                        )

            self._ptr = _cpp.models.QSRPDE(formula, data._ptr, level, params)

    def fit(self, lambda_=None, calibration_=None):
        r = None
        if calibration_ == None:
            self._ptr.fit(lambda_)
        else:
            r = self._ptr.fit_gcv(calibration_)
        self._f.set_coeff(self._ptr.f())
        if r != None:
            return r

    @property
    def f(self):
        return self._ptr.f()

    @property
    def beta(self):
        return self._ptr.beta()

    @property
    def fitted(self):
        return self._ptr.fitted()
