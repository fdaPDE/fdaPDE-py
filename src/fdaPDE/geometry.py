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

__all__ = ["Mesh"]


class Mesh(_cpp.geometry.Mesh):

    def __init__(self, nodes, cells, boundary):
        super().__init__(
            np.asarray(nodes, dtype=np.float64, order="F"),
            np.asarray(cells, dtype=np.int32, order="F"),
            np.asarray(boundary, dtype=np.int32, order="F").reshape(-1, 1),
        )

    def plot(self, ax=None, xlabel="", ylabel="", aspect=1, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:  ## create new panel if user didn't provide one
            _, ax = plt.subplots()

        ## set defaults
        if "color" not in kwargs:
            kwargs["color"] = "black"
        if "linewidth" not in kwargs:
            kwargs["linewidth"] = 0.5
        ## plot
        artists = ax.triplot(self.nodes[:, 0], self.nodes[:, 1], self.cells, **kwargs)
        ax.set_aspect(aspect)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        return ax

    def mapplot(
        self,
        boundary_nodes=None,
        domain_shape=None,
        map_location=None,
        zoom_start=7,
        tiles="CartoDB positron",
        domain_layer_name="domain",
        domain_color="black",
        domain_weight=2,
        domain_fill=True,
        domain_fill_color="grey",
        domain_fill_opacity=0.4,
        domain_show=True,
        mesh_layer_name="mesh",
        mesh_color="black",
        mesh_weight=0.8,
        mesh_opacity=0.8,
        mesh_show=True,
        map_height="500px",
    ):

        import folium
        from IPython.display import HTML

        if boundary_nodes is None and domain_shape is None:
            raise ValueError("Provide boundary_nodes or domain_shape.")
        if boundary_nodes is not None and domain_shape is not None:
            raise ValueError("Provide only one of boundary_nodes or domain_shape.")

        nodes = np.asarray(super().nodes)
        cells = np.asarray(super().cells)

        # set map location to mesh mid-point
        if map_location is None:
            map_location = [nodes[:, 1].mean(), nodes[:, 0].mean()]

        # create map
        m = folium.Map(
            location=map_location,
            zoom_start=zoom_start,
            tiles=tiles,
            width="100%",
            height=map_height,
        )

        # domain layer
        domain_fg = folium.FeatureGroup(name=domain_layer_name, show=domain_show)
        if boundary_nodes is not None:
            boundary_nodes = np.asarray(boundary_nodes)
            boundary_nodes = boundary_nodes[:, [1, 0]]
            folium.Polygon(
                locations=boundary_nodes,
                color=domain_color,
                weight=domain_weight,
                fill=domain_fill,
                fill_color=domain_fill_color,
                fill_opacity=domain_fill_opacity,
            ).add_to(domain_fg)
        else:
            folium.GeoJson(
                domain_shape,
                style_function=lambda x: {
                    "fillColor": domain_fill_color,
                    "color": domain_color,
                    "weight": domain_weight,
                    "fillOpacity": domain_fill_opacity,
                },
            ).add_to(domain_fg)
        domain_fg.add_to(m)

        # mesh layer
        mesh_fg = folium.FeatureGroup(name=mesh_layer_name, show=mesh_show)
        for tri in cells:
            points = [[nodes[tri[i], 1], nodes[tri[i], 0]] for i in range(3)]
            points.append(points[0])
            folium.PolyLine(
                points, color=mesh_color, weight=mesh_weight, opacity=mesh_opacity
            ).add_to(mesh_fg)
        mesh_fg.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)

        # display forcing height
        iframe_html = m._repr_html_()
        display(HTML(f"""
        <div style="width:100%; height:{map_height}; margin:0; padding:0;">
            <style>
                iframe {{
                    width: 100% !important;
                    height: {map_height} !important;
                }}
            </style>
            {iframe_html}
        </div>
        """))

    # def mapplot(
    #     self,
    #     boundary_nodes=None,
    #     map_location=None,
    #     zoom_start=6,
    #     tiles="cartodb positron",
    #     domain_layer_name="domain",
    #     domain_color="black",
    #     domain_weight=2,
    #     domain_fill=True,
    #     domain_fill_color="grey",
    #     domain_fill_opacity=0.4,
    #     domain_show=True,
    #     mesh_layer_name="mesh",
    #     mesh_color="black",
    #     mesh_weight=0.8,
    #     mesh_opacity=0.8,
    #     mesh_show=True,
    # ):

    #     import folium

    #     nodes = super().nodes
    #     cells = super().cells

    #     # set map location to mesh mid-point
    #     if map_location is None:
    #         map_location = [nodes[:, 1].mean(), nodes[:, 0].mean()]

    #     # create map
    #     m = folium.Map(location=map_location, zoom_start=zoom_start, tiles=tiles)

    #     # plot boundary
    #     if boundary_nodes is not None:
    #         domain_fg = folium.FeatureGroup(name=domain_layer_name, show=domain_show)
    #         boundary_nodes = np.asarray(boundary_nodes[:, [1, 0]])
    #         folium.Polygon(
    #             locations=boundary_nodes,
    #             color=domain_color,
    #             weight=domain_weight,
    #             fill=domain_fill,
    #             fill_color=domain_fill_color,
    #             fill_opacity=domain_fill_opacity,
    #         ).add_to(domain_fg)

    #         domain_fg.add_to(m)

    #     # plot mesh
    #     mesh_fg = folium.FeatureGroup(name=mesh_layer_name, show=mesh_show)
    #     for tri in cells:
    #         points = [
    #             [nodes[tri[0], 1], nodes[tri[0], 0]],
    #             [nodes[tri[1], 1], nodes[tri[1], 0]],
    #             [nodes[tri[2], 1], nodes[tri[2], 0]],
    #             [nodes[tri[0], 1], nodes[tri[0], 0]],
    #         ]

    #         folium.PolyLine(
    #             points, color=mesh_color, weight=mesh_weight, opacity=mesh_opacity
    #         ).add_to(mesh_fg)

    #     mesh_fg.add_to(m)

    #     folium.LayerControl(collapsed=False).add_to(m)
    #     return m

    def __str__(self):
        bbox = self.bbox
        out = [
            "2D mesh",
            f"Bounding box:   xmin: {bbox[0, 0]} ymin: {bbox[0, 1]} xmax: {bbox[1, 0]} ymax: {bbox[1, 1]}",
            f"Number of nodes: {self.n_nodes}",
            f"Number of cells: {self.n_cells}",
        ]
        return "\n".join(out)

    def info(self):
        print(self.__str__())
