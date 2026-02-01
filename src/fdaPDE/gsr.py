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

from ._gsr import cpp_gsr_2_2
from .formula import _formula
from .fe_function import _fe_function
import numpy as np

def gsr(formula, data, family, penalty = None):
    return _gsr(formula, data, family, penalty)

class _gsr:
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
            if not isinstance(f_obj, _fe_function):
                raise TypeError(f"Global symbol '{f_symbol}' is not a spatial field.")
        else:
            raise NameError(f"Global symbol '{f_symbol}' not found.")
        self._f = f_obj # store reference to field
        
        if penalty == None:
            self._ptr = cpp_gsr_2_2(formula, data, family, None)
        else:
            ptr_mesh = data._domain._ptr
            embed_dim = data._domain.embed_dim
            quad_nodes = ptr_mesh.quadrature_nodes()
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
                    params[name] = np.zeros((n_quad_nodes, size))
                else:
                    params[name] = np.tile(np.asarray(value).flatten(), (n_quad_nodes, 1))
                    
            self._ptr = cpp_gsr_2_2(formula, data, family, params)

    def fit(self, lambda_):
        self._ptr.fit(lambda_)
        self._f.set_coeff(self._ptr.f())

    @property
    def f(self):
        return self._ptr.f()

    @property
    def beta(self):
        return self._ptr.beta()
    
    @property
    def fitted(self):
        return self._ptr.fitted()
