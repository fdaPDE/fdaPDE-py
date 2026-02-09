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
        return self._ptr.grid_eval(np.asarray(locations))

    def plot(
        self,
        ax=None,
        xlabel="",
        ylabel="",
        title="",
        aspect=1,
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
        boundary_nodes=None,
        domain_shape=None,
        nx=200,
        ny=200,
        cmap_name="Blues",
        palette=None,
        opacity=1.0,
        layer_name="f",
        zoom_start=7,
        tiles="cartodb positron",
        epsg_map=4326,
        epsg_centroid=32617,
    ):

        import numpy as np
        import folium
        import matplotlib
        import matplotlib.colors as mcolors
        import branca.colormap as bc
        import geopandas as gpd
        from shapely.geometry import Polygon, Point
        from IPython.display import HTML
        
        # --------------------------------------------------
        # Input checks
        if boundary_nodes is None and domain_shape is None:
            raise ValueError("Provide either boundary_nodes or domain_shape.")

        if boundary_nodes is not None and domain_shape is not None:
            raise ValueError("Provide only one of boundary_nodes or domain_shape.")

        lon_nodes = self.mesh.nodes[:, 0]
        lat_nodes = self.mesh.nodes[:, 1]

        lat_lin = np.linspace(lat_nodes.min(), lat_nodes.max(), ny)
        lon_lin = np.linspace(lon_nodes.min(), lon_nodes.max(), nx)

        LAT, LON = np.meshgrid(lat_lin, lon_lin)
        grid_points = np.column_stack((LON.ravel(), LAT.ravel()))

        f_eval = np.asarray(self.eval(grid_points))
        f_mat = f_eval.reshape(ny, nx)

        if boundary_nodes is not None:
            boundary_nodes = np.asarray(boundary_nodes)
            if boundary_nodes.shape[1] != 2:
                raise ValueError("boundary_nodes must have shape (N, 2)")
            domain_geom = Polygon(boundary_nodes)
            map_center = [lat_nodes.mean(), lon_nodes.mean()]
        else:
            domain = domain_shape.copy()

            if domain.crs is None:
                domain = domain.set_crs(epsg=epsg_map)

            domain = domain.to_crs(epsg=epsg_map)
            domain_geom = domain.geometry.union_all()

            domain_utm = domain.to_crs(epsg=epsg_centroid)
            centroid = domain_utm.geometry.centroid.to_crs(epsg=epsg_map)
            map_center = [centroid.y.mean(), centroid.x.mean()]

        mask = np.array(
            [
                domain_geom.contains(Point(x, y))
                for x, y in zip(LON.ravel(), LAT.ravel())
            ]
        ).reshape(LAT.shape)

        f_mat[~mask] = np.nan

        f_grid = np.flipud(f_mat.T)

        vmin = np.nanmin(f_grid)
        vmax = np.nanmax(f_grid)

        # colormap handling
        if palette is not None:
            colormap = bc.LinearColormap(
                colors=palette, vmin=vmin, vmax=vmax, caption=layer_name
            )
            cmap_func = colormap
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            rgba_img = np.zeros((*f_grid.shape, 4))
            for i in range(f_grid.shape[0]):
                for j in range(f_grid.shape[1]):
                    val = f_grid[i, j]
                    if np.isnan(val):
                        rgba_img[i, j] = [0, 0, 0, 0]
                    else:
                        rgba_img[i, j] = mcolors.to_rgba(cmap_func(val))

            rgba_img = (rgba_img * 255).astype(np.uint8)

        else:
            cmap = matplotlib.colormaps[cmap_name]
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            rgba_img = cmap(norm(f_grid))
            rgba_img = (rgba_img * 255).astype(np.uint8)

            colormap = bc.LinearColormap(
                colors=[cmap(i) for i in np.linspace(0, 1, 256)],
                vmin=vmin,
                vmax=vmax,
                caption=layer_name,
            )

        bounds = [
            [lat_nodes.min(), lon_nodes.min()],
            [lat_nodes.max(), lon_nodes.max()],
        ]

        m = folium.Map(location=map_center, zoom_start=zoom_start, tiles=tiles)

        folium.raster_layers.ImageOverlay(
            image=rgba_img,
            bounds=bounds,
            opacity=opacity,
            name=layer_name,
            interactive=True,
        ).add_to(m)

        domain_fg = folium.FeatureGroup(name="domain", show=True)

        if boundary_nodes is not None:
            folium.Polygon(
                locations=boundary_nodes[:, [1, 0]], color="black", weight=2, fill=False
            ).add_to(domain_fg)
        else:
            folium.GeoJson(
                domain,
                style_function=lambda x: {
                    "fillColor": "none",
                    "color": "black",
                    "weight": 2,
                },
            ).add_to(domain_fg)

        domain_fg.add_to(m)

        colormap.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)

        html = m.get_root()._repr_html_()
        HTML(f"""
        <div style="
            width: 100%;
            height: 600px;
            margin: 0;
            padding: 0;
            overflow: hidden;
        ">
        {html}
        </div>
        """)

        return m
