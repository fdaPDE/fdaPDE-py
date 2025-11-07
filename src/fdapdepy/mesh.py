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
    def __init__(self, mesh, local_dim, embed_dim):
        self.__local_dim = local_dim
        self.__embed_dim = embed_dim
        self.__mesh = mesh

    def locate(self, locations) :
        return self.__mesh.locate(locations)

    def sample(self, n_samples, seed = None):
        if(seed == None):
            seed = -1 ## set random seed if not specified
        return self.__mesh.sample(n_samples, seed)
        
    def nodes(self):
        return self.__mesh.nodes()

    def edges(self):
        return self.__mesh.edges()

    def cells(self):
        return self.__mesh.cells()

    def boundary_nodes(self):
        return self.__mesh.boundary_nodes()

    def boundary_edges(self):
        return self.__mesh.boundary_edges()

    def n_nodes(self):
        return self.__mesh.n_nodes()
    
    def n_cells(self):
        return self.__mesh.n_cells()

    def n_edges(self):
        return self.__mesh.n_edges()

    def n_boundary_nodes(self):
        return self.__mesh.n_boundary_nodes()

    def n_boundary_edges(self):
        return self.__mesh.n_boundary_edges()

    def bbox(self):
        return self.__mesh.bbox()

    def measure(self):
        return self.__mesh.measure()

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
