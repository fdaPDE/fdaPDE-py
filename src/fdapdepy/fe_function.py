from ._fe_function import cpp_fe_function_2_2_p1

def fe_function(domain, fe_type="P1", coeff=None):
    """
    Create a finite element function object.

    Parameters
    ----------
    domain : triangulation object 
    type : type of basis functions ("P1", "P2", etc.).
    coeff : array-like or None Coefficients for the basis expansion.

    Returns
    -------
    FeFunction
        A Python object wrapping the corresponding C++ finite element function.
    """
    return FeFunction(domain=domain, type_=type_, coeff=coeff)

    
class FeFunction:
    def __init__(self, domain, fe_type, coeff=None):
        self._mesh = domain
        local_dim = domain.local_dim()
        embed_dim = domain.embed_dim()

        if local_dim == 2 and embed_dim == 2:
            self._fe_function = cpp_fe_function_2_2_p1(domain)

        self._type = fe_type

        n_dofs = self._fe_function.n_dofs()
        if coeff is not None:
            if len(coeff) != n_dofs:
                raise ValueError("Invalid coefficient vector dimensions.")
            self._fe_function.set_coeff(coeff)
        else:
            # Default to zero coefficients
            import numpy as np
            self._fe_function.set_coeff(np.zeros((n_dofs, 1)))

    def eval(self, locations):
        import numpy as np
        return self._fe_function.grid_eval(np.asarray(locations))

    def integral(self, marker=None):
        if marker is None:
            marker = -1
        return self._fe_function.cell_integrate_on(marker)

    def n_dofs(self):
        return self._fe_function.n_dofs()

    def l2_norm(self):
        return self._fe_function.l2_norm()

    def h1_norm(self):
        return self._fe_function.h1_norm()

    def l2_squared_norm(self):
        return self._fe_function.l2_squared_norm()

    def h1_squared_norm(self):
        return self._fe_function.h1_squared_norm()

    def coeff(self):
        return self._fe_function.coeff()

    def set_coeff(self, c):
        import numpy as np
        self._fe_function.set_coeff(np.asarray(c))

    def geometry(self):
        return self._mesh

    def plot(self, log_scale = True, boundary_nodes = None, cmap = "mako", aspect = 1, show = False):
        import matplotlib.pyplot as plt
        from matplotlib.tri import Triangulation
        import seaborn as sns
        import numpy as np

        """
        Plot the finite element function over its domain.

        Parameters
        ----------
        log_scale : bool
            If True, plot in log scale; if False, plot exp(f_fit).
        boundary_nodes : ndarray (m x 2), optional
            Coordinates of boundary nodes to draw the outline.
        cmap : str
            Name of the colour palette (default = "mako").
        aspect : float
            Aspect ratio of the axes (default = 1.4).
        show : bool
            If True, displays the plot; otherwise, returns (fig, ax).
        """

        triangulation = Triangulation(self._mesh.nodes()[:,0], self._mesh.nodes()[:,1], self._mesh.cells())
        f_fit = self.coeff().flatten()

        if not log_scale:
            f_fit = np.exp(f_fit)

        fig, ax = plt.subplots()

        tpc = ax.tripcolor(
            triangulation,
            f_fit, cmap=sns.color_palette(cmap, as_cmap=True)
        )

        if boundary_nodes is not None:
            ax.plot(
                np.concatenate((boundary_nodes[:, 0], [boundary_nodes[0, 0]])),
                np.concatenate((boundary_nodes[:, 1], [boundary_nodes[0, 1]])),
                color='black',
                linewidth=1
            )

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title("Log-scale" if log_scale else "Original scale")
        ax.set_aspect(aspect)

        cbar = fig.colorbar(tpc, ax=ax, fraction=0.03)
        cbar.set_label("log-rainfall" if log_scale else "rainfall", labelpad=10)

        if show:
            plt.show()
        else:
            return fig, ax



    

