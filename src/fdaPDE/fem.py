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
import numpy as np

__all__ = ["FeFunction"]


class FeFunction:
    def __init__(self, domain, fe_type="P1", coeff=None):
        self._domain = domain
        self._fe_type = fe_type
        local_dim = domain.dim[0]
        embed_dim = domain.dim[1]

        self._ptr = _cpp.fem.FeFunction(domain, 1 if fe_type == "P1" else 2)  # fix

        n_dofs = self._ptr.n_dofs
        if coeff is not None:
            if len(coeff) != n_dofs:
                raise ValueError("Invalid coefficient vector dimensions.")
            self._ptr.set_coeff(np.array(coeff, dtype=np.float64, order="C"))
        else:
            # default to zero coefficients
            self._ptr.set_coeff(np.zeros((n_dofs), dtype=np.float64, order="C"))

    @property
    def n_dofs(self):
        return self._ptr.n_dofs

    @property
    def l2_norm(self):
        return self._ptr.l2_norm

    @property
    def h1_norm(self):
        return self._ptr.h1_norm

    @property
    def l2_squared_norm(self):
        return self._ptr.l2_squared_norm

    @property
    def h1_squared_norm(self):
        return self._ptr.h1_squared_norm

    @property
    def coeff(self):
        return self._ptr.coeff

    @property
    def mesh(self):
        return self._domain

    def set_coeff(self, c):
        self._ptr.set_coeff(np.asarray(c))

    def eval(self, locations):
        return self._ptr.grid_eval(
            np.asarray(locations), self._domain.locate(locations)
        )

    def plot(
        self,
        ax=None,
        xlabel="",
        ylabel="",
        title="",
        aspect=1,
        log_scale=True,
        boundary_nodes=None,
        **kwargs,
    ):
        import matplotlib.pyplot as plt
        import matplotlib.tri as tri

        if ax is None:  ## create new panel if user didn't provide one
            _, ax = plt.subplots()

        ## build triangulation object
        triangulation = tri.Triangulation(
            self._domain.nodes[:, 0], self._domain.nodes[:, 1], self._domain.cells
        )
        f_fit = self.coeff.flatten()

        ## defaults
        if "cmap" not in kwargs:
            import seaborn as sns

            kwargs["cmap"] = sns.color_palette("mako", as_cmap=True)
        if not log_scale:
            f_fit = np.exp(f_fit)

        tpc = ax.tripcolor(triangulation, f_fit, **kwargs)
        ## plot domain boundary
        if boundary_nodes is not None:
            ax.plot(
                np.concatenate((boundary_nodes[:, 0], [boundary_nodes[0, 0]])),
                np.concatenate((boundary_nodes[:, 1], [boundary_nodes[0, 1]])),
                color="black",
                linewidth=1,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect(aspect)
        ## set colorbar
        cbar = ax.figure.colorbar(tpc, ax=ax, fraction=0.03)

        return ax

    def mapplot(
        self,
        boundary_nodes,
        nx=200,
        ny=200,
        cmap_name="Blues",
        opacity=1.0,
        layer_name="f",
        zoom_start=7,
        tiles="cartodb positron",
    ):

        import folium
        import matplotlib
        import matplotlib.colors as mcolors
        import branca.colormap as bc
        from folium.raster_layers import ImageOverlay
        from shapely.geometry import Point, Polygon
        
        # evaluate fe function on regular fine grid
        lat_nodes = self._domain.nodes[:, 1]
        lon_nodes = self._domain.nodes[:, 0]

        lat_lin = np.linspace(lat_nodes.min(), lat_nodes.max(), ny)
        lon_lin = np.linspace(lon_nodes.min(), lon_nodes.max(), nx)
        LAT, LON = np.meshgrid(lat_lin, lon_lin)
        grid_points = np.column_stack((LON.ravel(), LAT.ravel()))

        # evaluate
        f_eval = np.asarray(self._ptr.grid_eval(grid_points))
        f_mat = f_eval.reshape(ny, nx)

        boundary_nodes_latlon = np.asarray(boundary_nodes)[:, [1, 0]]
        domain_poly = Polygon(boundary_nodes_latlon)

        mask = np.array(
            [
                domain_poly.contains(Point(x, y))
                for x, y in zip(LAT.ravel(), LON.ravel())
            ]
        ).reshape(LAT.shape)

        f_mat[~mask] = np.nan

        f_grid = np.flipud(f_mat.T)

        vmin = np.nanmin(f_grid)
        vmax = np.nanmax(f_grid)

        cmap = matplotlib.colormaps[cmap_name]
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        rgba_img = cmap(norm(f_grid))
        rgba_img = (rgba_img * 255).astype(np.uint8)

        bounds = [
            [lat_nodes.min(), lon_nodes.min()],  # SW
            [lat_nodes.max(), lon_nodes.max()],  # NE
        ]

        m = folium.Map(
            location=[lat_nodes.mean(), lon_nodes.mean()],
            zoom_start=zoom_start,
            tiles=tiles,
        )

        ImageOverlay(
            image=rgba_img,
            bounds=bounds,
            opacity=opacity,
            name=layer_name,
            interactive=True,
        ).add_to(m)

        domain_fg = folium.FeatureGroup(name="domain", show=True)

        folium.Polygon(
            locations=boundary_nodes_latlon,
            color="black",
            weight=2,
            fill=False,
            name="domain",
        ).add_to(domain_fg)

        domain_fg.add_to(m)

        colormap = bc.LinearColormap(
            colors=[cmap(i) for i in np.linspace(0, 1, 256)],
            vmin=vmin,
            vmax=vmax,
            caption=layer_name,
        )

        colormap.add_to(m)

        folium.LayerControl().add_to(m)

        return m
