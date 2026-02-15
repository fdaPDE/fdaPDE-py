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

__all__ = ["SRPDE", "GSRPDE", "QSRPDE", "PPE", "fPCA", "GCV", "FeElliptic"]


def GCV(optimizer, edf="stochastic", mc_samples=100, seed=None):
    args = {"mc_samples": mc_samples, "seed": -1 if seed == None else seed}
    args.update(optimizer)
    return args

class FeElliptic:
    def __init__(self, K=None, b=None, c=None, u=None):
        self.K = K
        self.b = b
        self.c = c
        self.u = u

    def expand(self, mesh):
        """Expand PDE coefficients at quadrature nodes."""
        embed_dim = mesh.dim[1]

        quad_nodes = np.asarray(
            _cpp.fem.fe_simplex_2d_p1_quadrature(mesh)
        )
        n_q = quad_nodes.shape[0]

        sizes = {
            "K": embed_dim * embed_dim,
            "b": embed_dim,
            "c": 1,
            "u": 1,
        }

        def _expand_field(value, size):
            if value is None:
                arr = np.zeros((n_q, size))
            elif callable(value):
                arr = np.asarray(value(quad_nodes))
            else:
                arr = np.tile(np.asarray(value).ravel(), (n_q, 1))

            return arr if size > 1 else arr.ravel()

        return {
            name: _expand_field(getattr(self, name), size)
            for name, size in sizes.items()
        }

def _is_nonparametric_term(term, env):
    try:
        obj = env[term]
    except KeyError:
        return False
    return isinstance(obj, FeFunction)

    
def _is_formula_valid(formula, data, env):
    tmp = _formula(formula)
    
    if len(tmp.lhs) != 1:
        raise ValueError("Invalid formula: expected exactly one response term.")

    # lhs must be in data
    if tmp.lhs[0] not in data.colnames:
        raise ValueError(
            f"Invalid formula: term '{tmp.lhs[0]}' is not in data."
        )

    # rhs must be in data or a parametric term
    rhs = tmp.rhs
    unknowns = set(tmp.rhs) - set(data.colnames)
    for term in unknowns:
        if not _is_nonparametric_term(term, env):
            raise ValueError(
                f"Invalid formula: term '{term}' is not in data nor a nonparametric term."
            )

    # check that only one nonparametric term exists
    if len(unknowns) != 1:
        raise ValueError("Invaild formula: expected exactly one nonparametric term.")


def _extract_nonparametric_term(formula, data, env):
    tmp = _formula(formula)
    f_candidates = [v for v in tmp.rhs if v not in data.colnames]        
    return env[f_candidates[0]]


class SRPDE:
    def __init__(self, formula, data, penalty=None):
        import sys
        main_globals = sys.modules["__main__"].__dict__
        
        try:
            _is_formula_valid(formula, data, main_globals)
        except ValueError:
            raise
        
        self._f = _extract_nonparametric_term(formula, data, main_globals)
        
        if penalty is None:
            self._ptr = _cpp.models.SRPDE(formula, data._ptr, None)
            return

        params = penalty.expand(data.mesh)
        self._ptr = _cpp.models.SRPDE(formula, data._ptr, params)

    def fit(self, lambda_=None, calibration_=None):
        if calibration_ is None:
            self._ptr.fit(lambda_)
            result = None
        else:
            result = self._ptr.fit_gcv(calibration_)

        self._f.set_coeff(self._ptr.f())
        return result

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
    def __init__(self, formula, data, family, penalty=None):
        import sys
        main_globals = sys.modules["__main__"].__dict__
        
        try:
            _is_formula_valid(formula, data, main_globals)
        except ValueError:
            raise
        
        self._f = _extract_nonparametric_term(formula, data, main_globals)
        
        if penalty is None:
            self._ptr = _cpp.models.GSRPDE(formula, data._ptr, family, None)
            return

        params = penalty.expand(data.mesh)
        self._ptr = _cpp.models.GSRPDE(formula, data._ptr, family, params)

    def fit(self, lambda_=None, calibration_=None):
        if calibration_ is None:
            self._ptr.fit(lambda_)
            result = None
        else:
            result = self._ptr.fit_gcv(calibration_)

        self._f.set_coeff(self._ptr.f())
        return result

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
    def __init__(self, formula, data, level, penalty=None):
        import sys
        main_globals = sys.modules["__main__"].__dict__
        
        try:
            _is_formula_valid(formula, data, main_globals)
        except ValueError:
            raise
        
        self._f = _extract_nonparametric_term(formula, data, main_globals)
        
        if penalty is None:
            self._ptr = _cpp.models.QSRPDE(formula, data._ptr, level, None)
            return

        params = penalty.expand(data.mesh)
        self._ptr = _cpp.models.QSRPDE(formula, data._ptr, level, params)

    def fit(self, lambda_=None, calibration_=None):
        if calibration_ is None:
            self._ptr.fit(lambda_)
            result = None
        else:
            result = self._ptr.fit_gcv(calibration_)

        self._f.set_coeff(self._ptr.f())
        return result

    @property
    def f(self):
        return self._ptr.f()

    @property
    def beta(self):
        return self._ptr.beta()

    @property
    def fitted(self):
        return self._ptr.fitted()

class PPE:
    def __init__(self, data, penalty=None):
        self._ptr = _cpp.models.DEPDE(data._ptr, penalty)

    def fit(self, lambda_, optimizer_):
        return self._ptr.fit(lambda_, optimizer_)

    @property
    def density(self):
        return self._ptr.density()

    @property
    def log_density(self):
        return self._ptr.log_density()

    @property
    def fitted(self):
        return self._ptr.fitted()


class fPCA:
    def __init__(self, column, data):
        self._ptr = _cpp.models.fPCA(column, data._ptr)

    def fit(self, npc_, calibration_):
        self._ptr.fit(npc_, calibration_)

    @property
    def scores(self):
        return self._ptr.scores()

    @property
    def loadings(self):
        return self._ptr.loadings()

    @property
    def pcs(self):
        return self._ptr.pcs()

    @property
    def pcs_norm(self):
        return self._ptr.pcs_norm()

