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
from enum import IntEnum
import numpy as np

__all__ = ["GeoFrame"]


class layer_t(IntEnum):
    POINT = 0
    AREAL = 1


class data_t(IntEnum):
    FLT64 = 0
    FLT32 = 1
    INT64 = 2
    INT32 = 3
    BIN = 4
    STR = 5


class GeoFrame:
    def __init__(self, domain):
        local_dim = domain.dim[0]
        embed_dim = domain.dim[1]
        self._ptr = _cpp.GeoFrame(domain)
        self._layer_map = {}

    def insert(self, layer, layer_type, geo=None, data=None):
        if layer in self._layer_map.keys():
            raise ValueError(f"Layer {layer} already exists")

        # load data
        env = {"int_data": {}, "dbl_data": {}, "str_data": {}}

        def load(a, b, colname):
            """Copies `a` inside `b[colname]`, dispatching by dtype."""
            arr = np.asarray(a)
            # mapping from dtype to target key
            dtype_map = {
                np.floating: "dbl_data",
                np.integer: "int_data",
                np.str_: "str_data",
            }
            # load data
            for dtype_class, key in dtype_map.items():
                if np.issubdtype(arr.dtype, dtype_class):
                    b[key][colname] = arr
                    return

        if data is not None:
            if isinstance(data, np.ndarray):
                n_col = 1 if data.ndim == 1 else data.shape[1]
                colnames = [f"V{i+1}" for i in range(n_col)]  # defaults column names
                # normalize data to 2D ndarray
                data = np.atleast_2d(data).T if data.ndim == 1 else data
                iterator = zip(colnames, (data[:, i] for i in range(n_col)))
            elif hasattr(data, "columns") and hasattr(
                data, "to_numpy"
            ):  # duck typing, but do with lazy loading
                iterator = ((c, np.array(data[c])) for c in data.columns)
            else:
                raise TypeError("Unsupported data type.")

            skip_cols = []
            if isinstance(geo, (str, list)):
                skip_cols = [geo] if isinstance(geo, str) else geo

            for colname, col in iterator:
                if colname in skip_cols:
                    continue
                load(col, env, colname)

        # handle geometry information
        if geo is None:
            if layer_type != "point":
                raise ValueError("Missing geometry.")
            # self._ptr.insert_scalar_point_layer_nodes(layer, 0, env)
            # self._layer_map[layer] = "point"
        else:
            # geometry could be column names or direct coordinates
            if isinstance(geo, (str, list)):
                geo_ = (
                    np.asarray(data[geo])
                    if hasattr(data, "__getitem__")
                    else np.array(geo)
                )
            else:
                geo_ = np.asarray(geo)

            if layer_type == "point":
                self._ptr.point_insert_layer(
                    layer, np.array(geo_, dtype=np.float64, order="F"), env
                )  # qui viene fatta una copia, perchè eigen vuole col-major.....
                self._layer_map[layer] = "point"

            if layer_type == "areal":
                self._ptr.areal_insert_layer(
                    layer, np.array(geo_, dtype=np.float64, order="F"), env
                )
                self._layer_map[layer] = "areal"

    def load_shp(self, layer, filename):
        self._ptr.load_shp(layer, filename)
        self._layer_map[layer] = "areal"

    def __str__(self):
        """Print method for the geoframe class."""
        n_layers = len(self._layer_map)
        layer_names = list(self._layer_map.keys())
        bbox = self._ptr.mesh.bbox
        n_nodes = self._ptr.mesh.n_nodes
        n_cells = self._ptr.mesh.n_cells

        out = []
        out.append(f"Geoframe with {n_layers} layer{'s' if n_layers != 1 else ''}")
        out.append(
            f"Bounding box:   xmin: {bbox[0, 0]} ymin: {bbox[0, 1]} xmax: {bbox[1, 0]} ymax: {bbox[1, 1]}",
        )
        out.append(f"Number of nodes: {n_nodes}")
        out.append(f"Number of cells: {n_cells}\n")

        if n_layers > 0:
            for name in layer_names:
                out.append(f"Layer: {name}")
                ncols = self._ptr.cols(name)
                nrows = self._ptr.rows(name)
                if ncols == 0:
                    out.append("Type:  POINT PATTERN")
                    out.append(f"Dims:  {nrows}, {self._ptr.mesh.dim[1]}")
                else:
                    layer_type = self._layer_map[name]
                    out.append(f"Type:  {layer_type.upper()}")
                    out.append(f"Dims:  {nrows}, {ncols}")

                    # show first few data rows
                    n_preview = min(6, nrows)
                    out.append(f"First {n_preview} data rows:")

                    if layer_type == "areal":
                        layer = _areal_layer(self, name)
                        out.append(str(layer))
                        out.append("")
                    if layer_type == "point":
                        layer = _point_layer(self, name)
                        out.append(str(layer))
                        out.append("")

        return "\n".join(out)

    def __getitem__(self, layer_name):
        if layer_name not in self._layer_map:
            raise KeyError(f"Layer '{layer_name}' not found.")

        layer_type = self._layer_map[layer_name]
        if layer_type == "areal":
            return _areal_layer(self, layer_name)
        if layer_type == "point":
            return _point_layer(self, layer_name)

    @property
    def n_layers(self):
        return len(self._layer_map.keys())

    @property
    def names(self):
        return list(self._layer_map.keys())

    @property
    def mesh(self):
        return self._ptr.mesh

    @property
    def colnames(self):
        return [x for xs in [self.__getitem__(layer).colnames for layer in self.names] for x in xs]


## low-level typed dispatch logic
def _gf_cpp_access(x, rows, col):
    ptr = x._ptr
    layer_name = x.name
    dtype = ptr.dtype(layer_name, col)

    access_map = {
        data_t.FLT64: ptr.flt64_access,
        # data_t.FLT32: ptr.flt32_access,
        # data_t.INT64: ptr.int64_access,
        data_t.INT32: ptr.int32_access,
        data_t.STR: ptr.str_access,
    }
    return access_map[dtype](layer_name, rows, col)


def _gf_cpp_assign(
    x, rows, col, value
):  # qui col può essere solo una stringa o un numero
    import numpy as np

    ptr = x._ptr
    layer_name = x.name
    dtype = ptr.dtype(layer_name, col)
    value = np.asarray(value)

    assign_map = {
        data_t.FLT64: (ptr.flt64_assign, np.float64),
        # data_t.FLT32: (ptr.flt32_assign, np.float32),
        # data_t.INT64: (ptr.int64_assign, np.int64  ),
        data_t.INT32: (ptr.int32_assign, np.int32),
        data_t.STR: (ptr.str_assign, np.str_),
    }
    assign, typ = assign_map[dtype]
    # if scalar, keep scalar, otherwise np-cast
    casted_value = value.astype(typ).item() if value.ndim == 0 else value.astype(typ)
    assign(layer_name, rows, col, casted_value)


class _data_layer:
    def __init__(self, geoframe, name: str):
        self._geoframe = geoframe
        self._ptr = geoframe._ptr
        self._name = name

    def _get(self, rows=None, cols=None):
        # subsetting rows
        if rows is None:
            rows = list(range(self._ptr.rows(self._name)))
        else:
            if (
                isinstance(rows, (list, tuple, np.ndarray))
                and np.array(rows).dtype == bool
            ):
                rows = list(np.where(rows)[0])  # apply boolean filter
            else:
                rows = [rows] if isinstance(rows, int) else rows
                rows = [r for r in rows]
        # subsetting columns
        if cols is None:
            cols = self._ptr.colnames(self._name)
        else:
            if not all(isinstance(c, str) for c in cols):
                all_cols = self._ptr.colnames(self._name)
                cols = [all_cols[c] for c in cols]
            else:
                cols = cols if isinstance(cols, list) else [cols]

        ptr = _cpp.GeoFrame(self._ptr, self._name, rows, cols)
        gf = GeoFrame.__new__(GeoFrame)  # create geoframe bypassing __init__
        gf._ptr = ptr
        gf._layer_map = {self._name: self._geoframe._layer_map[self._name]}
        return gf

    def _set(self, rows, col, value):
        dtype = self._ptr.dtype(self._name, col)
        # prepare row subsetting vector
        if rows is None:
            rows = list(range(self._ptr.rows(self._name)))
        else:
            rows = [rows] if isinstance(rows, int) else rows
            rows = [r for r in rows]
        # broadcast scalar values
        if np.isscalar(value):
            value = [value] * len(rows)
        _gf_cpp_assign(self, rows, col, value)

    @property
    def name(self):
        return self._name

    @property
    def rows(self):
        return self._ptr.rows(self._name)

    @property
    def cols(self):
        return len(self._ptr.colnames(self._name))

    @property
    def colnames(self):
        return self._ptr.colnames(self._name)

    def col(self, name):
        return _gf_cpp_access(self, range(self.rows), name)

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            i = None if isinstance(i, slice) else i
            j = None if isinstance(j, slice) else j
            return self._get(i, j)

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            i = None if isinstance(i, slice) else i
            j = None if isinstance(j, slice) else j
            return self._set(i, j, value)


class _point_layer(_data_layer):

    def __str__(self):
        out = []
        colnames = self._ptr.colnames(self._name)
        nrows = min(6, self._ptr.rows(self._name))
        locations = self.geometry[:nrows]

        # prepare display
        # first column display geometrical information
        geo = [
            f"({coord[0]:.6f}, {coord[1]:.6f})" for coord in locations
        ]  # do not generalize with embed_dim dimension
        geo_column = ["", "<POINT>"] + geo
        out.append(geo_column)
        # display data column
        for colname in colnames:
            v = _gf_cpp_access(self, range(nrows), colname)
            dtype = self._ptr.dtype(self._name, colname)

            dtype_map = {
                data_t.FLT64: "<flt64>",
                data_t.FLT32: "<flt32>",
                data_t.INT64: "<int64>",
                data_t.INT32: "<int32>",
                data_t.BIN: "<bin>",
                data_t.STR: "<str>",
            }
            type_str = dtype_map.get(dtype, "<unk>")
            formatted_values = [f"{val:.6f}" for val in v]
            column = [colname, type_str] + formatted_values
            out.append(column)

        # column alignment
        col_widths = [max(len(s) for s in col) + 1 for col in out]
        lines = []
        # header
        header = "".join(
            col.rjust(w) for col, w in zip([c[0] for c in out], col_widths)
        )
        lines.append(header)
        type_line = "".join(
            f"\033[0;31m{c[1].rjust(w)}\033[0m" for c, w in zip(out, col_widths)
        )
        lines.append(type_line)
        # send to output stream
        for i in range(2, len(out[0])):
            row_line = "".join(out[j][i].rjust(col_widths[j]) for j in range(len(out)))
            lines.append(row_line)

        return "\n".join(lines)

    def plot(
        self, ax=None, xlabel="", ylabel="", aspect=1, boundary_nodes=None, **kwargs
    ):
        import matplotlib.pyplot as plt
        import math

        colnames = self._ptr.colnames(self._name)
        ncols = len(colnames)
        locations = self.geometry

        ## defaults
        if "cmap" not in kwargs:
            import seaborn as sns

            kwargs["cmap"] = sns.color_palette("mako", as_cmap=True)
        if "s" not in kwargs:
            kwargs["s"] = 1
        if "edgecolor" not in kwargs:
            kwargs["edgecolor"] = "none"

        # compute grid layout
        ncols_grid = int(math.ceil(math.sqrt(ncols)))
        nrows_grid = int(math.ceil(ncols / ncols_grid))

        if ax is None:  ## create new panel if user didn't provide one
            fig, ax = plt.subplots(nrows_grid, ncols_grid)

        # normalize ax to ndarray
        if not isinstance(ax, np.ndarray):
            ax = np.array([ax])
        else:
            ax = ax.flatten()

        # plot each column
        xlabel = [xlabel] if isinstance(xlabel, str) else xlabel
        ylabel = [ylabel] if isinstance(ylabel, str) else ylabel
        for i, colname in enumerate(colnames):
            # data scatter
            sc = ax[i].scatter(
                x=locations[:, 0], y=locations[:, 1], c=self.col(colname), **kwargs
            )

            ## plot domain boundary
            if boundary_nodes is not None:
                ax[i].plot(
                    np.concatenate((boundary_nodes[:, 0], [boundary_nodes[0, 0]])),
                    np.concatenate((boundary_nodes[:, 1], [boundary_nodes[0, 1]])),
                    color="black",
                    linewidth=1,
                )
            ax[i].set_title(colname)
            # ax[i].set_xlabel(xlabel[i]) ##################### check
            # ax[i].set_ylabel(ylabel[i])
            ax[i].set_aspect(aspect)

            # set colorbar
            cbar = ax[i].figure.colorbar(sc, ax=ax[i], fraction=0.03)

        # hide unused axes
        for j in range(i + 1, len(ax)):
            ax[j].set_visible(False)

        fig.tight_layout(pad=2)
        return ax

    def mapplot(
        self,
        colname,
        boundary_nodes=None,
        zoom_start=6,
        tiles="cartodb positron",
        cmap_name="viridis",
        marker_radius=6,
        marker_weight=1,
        marker_opacity=0.9,
        domain_color="black",
        domain_weight=2,
        domain_fill=False,
    ):

        import folium
        import matplotlib
        import matplotlib.colors as mcolors
        import branca.colormap as bc

        # get coordinates and values
        coords = self.geometry
        values = np.array(self._geoframe[self._name].col(colname))

        map_center = [coords[:, 1].mean(), coords[:, 0].mean()]
        m = folium.Map(location=map_center, zoom_start=zoom_start, tiles=tiles)

        # plot domain boundary
        if boundary_nodes is not None:
            boundary_nodes = np.asarray(boundary_nodes)[:, [1, 0]]  # lat, lon
            domain_fg = folium.FeatureGroup(name="domain", show=True)
            folium.Polygon(
                locations=boundary_nodes,
                color=domain_color,
                weight=domain_weight,
                fill=domain_fill,
            ).add_to(domain_fg)

            domain_fg.add_to(m)

        # colormap
        cmap = matplotlib.colormaps[cmap_name]
        norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
        colormap = bc.LinearColormap(
            colors=[cmap(i) for i in np.linspace(0, 1, 256)],
            vmin=values.min(),
            vmax=values.max(),
            caption=colname,
        )

        colormap.add_to(m)

        # plot data points
        data_fg = folium.FeatureGroup(name=colname, show=True)

        for i in range(len(values)):
            color = mcolors.to_hex(cmap(norm(values[i])))
            folium.CircleMarker(
                location=[coords[i, 1], coords[i, 0]],
                radius=marker_radius,
                color="black",
                weight=marker_weight,
                fill=True,
                fill_color=color,
                fill_opacity=marker_opacity,
                popup=f"{colname}: {values[i]}",
            ).add_to(data_fg)

            data_fg.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        return m

    @property
    def geometry(self):
        return self._ptr.point_coordinates(self._name)


class _areal_layer(_data_layer):

    @property
    def geometry(self):
        return self._ptr.areal_polygons(self._name)

    def __str__(self):
        out = []
        colnames = self._ptr.colnames(self._name)
        nrows = min(6, self._ptr.rows(self._name))
        locations = self.geometry[:nrows]

        # prepare display
        # first column display geometrical information
        geo = []
        for polygon in locations:
            nodes = polygon["nodes"]
            geo = geo + [
                f"[({nodes[0, 0]:.2f}, {nodes[0, 1]:.2f}), ..., ({nodes[nodes.shape[0] - 1, 0]:.2f}, {nodes[nodes.shape[0] - 1, 1]:.2f})]"
            ]
        geo_column = ["", "<POLYGON>"] + geo
        out.append(geo_column)
        # display data column
        for colname in colnames:
            v = _gf_cpp_access(self, range(nrows), colname)
            dtype = self._ptr.dtype(self._name, colname)

            dtype_map = {
                data_t.FLT64: "<flt64>",
                data_t.FLT32: "<flt32>",
                data_t.INT64: "<int64>",
                data_t.INT32: "<int32>",
                data_t.BIN: "<bin>",
                data_t.STR: "<str>",
            }
            type_str = dtype_map.get(dtype, "<unk>")
            formatted_values = [
                val if isinstance(val, str) else f"{val:.6f}" for val in v
            ]
            column = [colname, type_str] + formatted_values
            out.append(column)

        # column alignment
        col_widths = [max(len(s) for s in col) + 1 for col in out]
        lines = []
        # header
        header = "".join(
            col.rjust(w) for col, w in zip([c[0] for c in out], col_widths)
        )
        lines.append(header)
        type_line = "".join(
            f"\033[0;31m{c[1].rjust(w)}\033[0m" for c, w in zip(out, col_widths)
        )
        lines.append(type_line)
        # send to output stream
        for i in range(2, len(out[0])):
            row_line = "".join(out[j][i].rjust(col_widths[j]) for j in range(len(out)))
            lines.append(row_line)

        return "\n".join(lines)
