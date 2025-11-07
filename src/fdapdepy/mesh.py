from ._mesh import cpp_triangulation_2_2

def triangulation(nodes, cells, boundary):
    local_dim = cells.shape[1] - 1
    embed_dim = nodes.shape[1]
    ## instantiate cpp backend
    data = {
        "nodes": nodes,
        "cells": cells,
        "boundary": boundary
    }
    cpp_backend = None
    if (local_dim == 2 and embed_dim == 2):
        cpp_backend = cpp_triangulation_2_2(data)
        
    return __triangulation(cpp_backend, local_dim, embed_dim)

class __triangulation:
    def __init__(self, cpp_backend, local_dim, embed_dim):
        self.__local_dim = local_dim
        self.__embed_dim = embed_dim
        self._cpp_backend = cpp_backend

    def local_dim(self): return self.__local_dim
    def embed_dim(self): return self.__embed_dim    
        
    def locate(self, locations):
        return self._cpp_backend.locate(locations)

    def sample(self, n_samples, seed = None):
        if(seed == None):
            seed = -1 ## set random seed if not specified
        return self._cpp_backend.sample(n_samples, seed)
        
    def nodes(self):
        return self._cpp_backend.nodes()

    def edges(self):
        return self._cpp_backend.edges()

    def cells(self):
        return self._cpp_backend.cells()

    def boundary_nodes(self):
        return self._cpp_backend.boundary_nodes()

    def boundary_edges(self):
        return self._cpp_backend.boundary_edges()

    def n_nodes(self):
        return self._cpp_backend.n_nodes()
    
    def n_cells(self):
        return self._cpp_backend.n_cells()

    def n_edges(self):
        return self._cpp_backend.n_edges()

    def n_boundary_nodes(self):
        return self._cpp_backend.n_boundary_nodes()

    def n_boundary_edges(self):
        return self._cpp_backend.n_boundary_edges()

    def bbox(self):
        return self._cpp_backend.bbox()

    def measure(self):
        return self._cpp_backend.measure()

    def plot(self, ax = None, xlabel = "", ylabel = "", aspect = 1, show = False, **kwargs):
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
            self.nodes()[:,0], self.nodes()[:,1], self.cells(),
            **kwargs
        )
        ax.set_aspect(aspect)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        if show:
            plt.show()

        return ax
    
    def __str__(self):
        bbox = self.bbox()
        
        out = [
            "2D triangulation",
            f"Bounding box:   xmin: {bbox[0, 0]} ymin: {bbox[0, 1]} xmax: {bbox[1, 0]} ymax: {bbox[1, 1]}",
            f"Number of nodes: {self.n_nodes()}",
            f"Number of cells: {self.n_cells()}",
        ]

        return "\n".join(out)
    
    def info(self):
        print(self.__str__())
