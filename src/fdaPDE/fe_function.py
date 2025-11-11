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

from ._fe_function import cpp_fe_function_2_2_p1
import numpy as np

def fe_function(domain, fe_type = "P1", coeff = None):
    return _fe_function(domain = domain, fe_type = fe_type, coeff = coeff)
    
class _fe_function:
    def __init__(self, domain, fe_type, coeff = None):
        self._domain = domain
        self._fe_type = fe_type
        local_dim = domain.local_dim
        embed_dim = domain.embed_dim

        if local_dim == 2 and embed_dim == 2 and fe_type == "P1":
            self._ptr = cpp_fe_function_2_2_p1(domain)

        n_dofs = self._ptr.n_dofs()
        if coeff is not None:
            if len(coeff) != n_dofs:
                raise ValueError("Invalid coefficient vector dimensions.")
            self._ptr.set_coeff(coeff)
        else:
            self._ptr.set_coeff(np.zeros((n_dofs, 1))) # default to zero coefficients

    @property
    def n_dofs(self):
        return self._ptr.n_dofs()

    @property
    def l2_norm(self):
        return self._ptr.l2_norm()

    @property
    def h1_norm(self):
        return self._ptr.h1_norm()

    @property
    def l2_squared_norm(self):
        return self._ptr.l2_squared_norm()

    @property
    def h1_squared_norm(self):
        return self._ptr.h1_squared_norm()

    @property
    def coeff(self):
        return self._ptr.coeff()

    @property
    def geometry(self):
        return self._domain

    def set_coeff(self, c):
        self._ptr.set_coeff(np.asarray(c))
    
    def eval(self, locations):
        return self._ptr.grid_eval(np.asarray(locations))

    def integral(self, marker = None):
        if marker is None:
            marker = -1
        return self._ptr.cell_integrate_on(marker)
    
    def plot(self, ax = None, xlabel = "", ylabel = "", title = "", aspect = 1, log_scale = True, boundary_nodes = None,
             **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.tri as tri

        if ax is None: ## create new panel if user didn't provide one
            _, ax = plt.subplots()
        
        ## build triangulation object
        triangulation = tri.Triangulation(self._domain.nodes[:,0], self._domain.nodes[:,1], self._domain.cells)
        f_fit = self.coeff.flatten()

        ## defaults
        if "cmap" not in kwargs:
            import seaborn as sns
            kwargs["cmap"] = sns.color_palette("mako", as_cmap = True)
        if not log_scale:
            f_fit = np.exp(f_fit)
        
        tpc = ax.tripcolor(
            triangulation, f_fit,
            **kwargs
        )
        ## plot domain boundary
        if boundary_nodes is not None:
            ax.plot(
                np.concatenate((boundary_nodes[:, 0], [boundary_nodes[0, 0]])),
                np.concatenate((boundary_nodes[:, 1], [boundary_nodes[0, 1]])),
                color = "black",
                linewidth = 1
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect(aspect)
        ## set colorbar
        cbar = ax.figure.colorbar(tpc, ax = ax, fraction = 0.03)

        return ax
