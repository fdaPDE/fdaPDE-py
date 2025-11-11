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

from ._mesh import cpp_triangulation_2_2

def triangulation(nodes, cells, boundary):
    local_dim = cells.shape[1] - 1
    embed_dim = nodes.shape[1]
    ## instantiate cpp backend
    data = {"nodes": nodes, "cells": cells, "boundary": boundary}
    ptr = None
    if (local_dim == 2 and embed_dim == 2):
        ptr = cpp_triangulation_2_2(data)
        
    return _triangulation(ptr, local_dim, embed_dim)

class _triangulation:
    def __init__(self, ptr, local_dim, embed_dim):
        self._local_dim = local_dim
        self._embed_dim = embed_dim
        self._ptr = ptr

    @property
    def local_dim(self):
        return self._local_dim

    @property
    def embed_dim(self):
        return self._embed_dim
        
    @property
    def nodes(self):
        return self._ptr.nodes()

    @property
    def edges(self):
        return self._ptr.edges()

    @property
    def cells(self):
        return self._ptr.cells()

    @property
    def boundary_nodes(self):
        return self._ptr.boundary_nodes()

    @property
    def boundary_edges(self):
        return self._ptr.boundary_edges()

    @property
    def n_nodes(self):
        return self._ptr.n_nodes()

    @property
    def n_cells(self):
        return self._ptr.n_cells()

    @property
    def n_edges(self):
        return self._ptr.n_edges()

    @property
    def n_boundary_nodes(self):
        return self._ptr.n_boundary_nodes()

    @property
    def n_boundary_edges(self):
        return self._ptr.n_boundary_edges()

    @property
    def bbox(self):
        return self._ptr.bbox()

    @property
    def measure(self):
        return self._ptr.measure()

    def locate(self, locations):
        return self._ptr.locate(locations)

    def sample(self, n_samples, seed = None):
        if(seed == None):
            seed = -1 ## set random seed if not specified
        return self._ptr.sample(n_samples, seed)
    
    def plot(self, ax = None, xlabel = "", ylabel = "", aspect = 1, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None: ## create new panel if user didn't provide one
            _, ax = plt.subplots()

        ## set defaults
        if "color" not in kwargs:
            kwargs["color"] = "black"
        if "linewidth" not in kwargs:
            kwargs["linewidth"] = 0.5
        ## plot    
        artists = ax.triplot(
            self.nodes[:,0], self.nodes[:,1], self.cells,
            **kwargs
        )
        ax.set_aspect(aspect)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        return ax
    
    def __str__(self):
        bbox = self.bbox
        out = [
            "2D triangulation",
            f"Bounding box:   xmin: {bbox[0, 0]} ymin: {bbox[0, 1]} xmax: {bbox[1, 0]} ymax: {bbox[1, 1]}",
            f"Number of nodes: {self.n_nodes}",
            f"Number of cells: {self.n_cells}",
        ]
        return "\n".join(out)

    def info(self):
        print(self.__str__())
