from ._geoframe import cpp_geoframe_2_2
from enum import IntEnum

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

def geoframe(domain):
    local_dim = domain.local_dim()
    embed_dim = domain.embed_dim()

    cpp_backend = None
    if (local_dim == 2 and embed_dim == 2):
        cpp_backend = cpp_geoframe_2_2(domain)
    
    return _geoframe(cpp_backend, domain)

class _geoframe:
    def __init__(self, cpp_backend, domain):
        self._cpp_backend = cpp_backend
        self._domain = domain
        self._layer_map = {}

    def insert(self, layer, typ, geo = None, data = None):
        if layer in self._cpp_backend.laynames():
            raise ValueError(f"Layer {layer} already exists")

        # load data in homogeneous structure
        env = {"int_data": {}, "dbl_data": {}, "str_data": {}}
        def load(a, b, colname):
            """Copies 'a' inside 'b', depending on a's type"""
            import numpy as np
            if isinstance(a, (float, np.floating, np.ndarray)) and np.issubdtype(np.array(a).dtype, np.floating):
                b["dbl_data"][colname] = np.asarray(a)
            elif isinstance(a, (int, np.integer, np.ndarray)) and np.issubdtype(np.array(a).dtype, np.integer):
                b["int_data"][colname] = np.asarray(a)
            elif isinstance(a, (str, np.str_)) or (isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.str_)):
                b["str_data"][colname] = np.asarray(a)

        if data is not None:
            import numpy as np
            import pandas as pd

            if isinstance(data, (np.ndarray, pd.DataFrame)):
                if isinstance(data, np.ndarray):
                    n_col = 1 if data.ndim == 1 else data.shape[1]
                    colnames = [f"V{i+1}" for i in range(n_col)] # defaults column names
                    # normalize data to 2D ndarray
                    data = np.atleast_2d(data).T if data.ndim == 1 else data
                else:
                    n_col = data.shape[1]
                    # use dataframe colnames or set to default
                    colnames = data.columns.tolist() or [f"V{i+1}" for i in range(n_col)]

                # load data
                for i in range(n_col):
                    colname = colnames[i]
                    if isinstance(geo, (str, list)):
                        skip_cols = [geo] if isinstance(geo, str) else geo
                        if colname not in skip_cols:
                            load(data[:, i] if isinstance(data, np.ndarray) else data.iloc[:, i].to_numpy(),
                                 env, colname)
                    else:
                        load(data[:, i] if isinstance(data, np.ndarray) else data.iloc[:, i].to_numpy(),
                             env, colname)

        # handle geometry information
        if geo is None:
            if typ != "point":
                raise ValueError("Missing geometry.")            
            self._cpp_backend.insert_scalar_point_layer_nodes(layer, 0, env)
            self._layer_map[layer] = "point"
        else:
            import numpy as np

            # geometry could be column names or direct coordinates
            if isinstance(geo, (str, list)):
                geo_ = np.asarray(data[geo]) if hasattr(data, "__getitem__") else np.array(geo)
            else:
                geo_ = np.asarray(geo)

            if typ == "point":
                self._cpp_backend.insert_scalar_point_layer(layer, geo_, env)
                self._layer_map[layer] = "point"

            elif typ == "areal":
                self._cpp_backend.insert_scalar_areal_layer(layer, geo_, env)
                self._layer_map[layer] = "areal"

            else:
                raise ValueError(f"Unknown layer type: {type_}")

    def __str__(self):
        """Print method for the geoframe class."""
        n_layers = len(self._layer_map)
        layer_names = list(self._layer_map.keys())
        bbox = self._domain.bbox()
        n_nodes = self._domain.n_nodes()
        n_cells = self._domain.n_cells()

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
                ncols = self._cpp_backend.cols(name)
                nrows = self._cpp_backend.rows(name)
                if ncols == 0:
                    out.append("Type:  POINT PATTERN")
                    out.append(f"Dims:  {nrows}, {self._domain.embed_dim()}")
                else:
                    layer_type = self._layer_map[name]
                    out.append(f"Type:  {layer_type.upper()}")
                    out.append(f"Dims:  {nrows}, {ncols}")

                    # show first few data rows
                    n_preview = min(6, nrows)
                    out.append(f"First {n_preview} data rows:")

                    if layer_type == "areal":
                        layer = _areal_layer(self._cpp_backend, self._domain, name, "areal") ## ma sà già che è areale, perchè glielo devo dire?
                        out.append(str(layer))
                    elif layer_type == "point":
                        layer = _point_layer(self._cpp_backend, self._domain, name, "point")
                        out.append(str(layer))
                        out.append("")

        return "\n".join(out)

    def __getitem__(self, layer_name: str):
        if layer_name not in self._layer_map:
            raise KeyError(f"Layer '{layer_name}' not found.")

        layer_type = self._layer_map[layer_name]

        if layer_type == "areal":
            return _areal_layer(self._cpp_backend, self._domain, layer_name, "areal")
        elif layer_type == "point":
            return _point_layer(self._cpp_backend, self._domain, layer_name, "point")

## low-level typed dispatch logic
def _gf_cpp_access(x, rows, col):
    cpp_backend = x._cpp_backend
    layer_name = x.name
    dtype = cpp_backend.dtype(layer_name, col)

    access_map = {
        data_t.FLT64: cpp_backend.flt64_access,
        data_t.FLT32: cpp_backend.flt32_access,
        data_t.INT64: cpp_backend.int64_access,
        data_t.INT32: cpp_backend.int32_access,
        #data_t.BIN:   cpp_backend.bin_access,
        data_t.STR:   cpp_backend.str_access,
    }
    return access_map[dtype](layer_name, rows, col)
        
class _data_layer:
    def __init__(self, cpp_backend, mesh, name: str, typ: str):
        self._cpp_backend = cpp_backend
        self._mesh = mesh
        self._name = name
        if typ not in ("point", "areal"):
            raise ValueError("Invalid type: must be 'point' or 'areal'.")
        self._typ = typ

    def get(self, rows = None, cols = None):
        """Subset the layer, returning a new GeoFrame-like object."""
        # Handle row indices
        if rows is None:
            rows = list(range(self._cpp_backend.rows(self._name)))
        elif isinstance(rows, (list, tuple, np.ndarray)) and np.array(rows).dtype == bool:
            rows = list(np.where(rows)[0])
        else:
            rows = [r for r in rows]

        # Handle column names
        if cols is None:
            cols = self._cpp_backend.colnames(self._name)
        elif not all(isinstance(c, str) for c in cols):
            all_cols = self._cpp_backend.colnames(self._name)
            cols = [all_cols[c] for c in cols]
        else:
            cols = list(cols)

        new_ptr = cpp_geoframe_2_2(self._cpp_backend, self._name, rows, cols)
        new_gf = _geoframe.__new__(_geoframe) ## create geoframe bypassing __init__
        new_gf._ptr = new_ptr
        new_gf._layer_map = {self._name: self._type}
        return new_gf

    # def set(self, rows, col, value):
    #     """Set values in the layer."""
    #     dtype = self._ptr.dtype(self._name, col)

    #     # Prepare row indices
    #     if rows is None:
    #         rows = list(range(self._ptr.rows(self._name)))
    #     else:
    #         rows = [r for r in rows]

    #     # Broadcast scalar values
    #     if np.isscalar(value):
    #         value = [value] * len(rows)

    #     _gf_cpp_assign(self, rows, col, value)

    @property
    def name(self) -> str:
        return self._name
    @property
    def rows(self) -> int:
        return self._cpp_backend.rows(self._name)
    @property
    def cols(self) -> int:
        return len(self._cpp_backend.colnames(self._name))
    @property
    def colnames(self):
        return self._cpp_backend.colnames(self._name)

    def col(self, name) :
        return _gf_cpp_access(self, range(self.rows), name)
        
class _point_layer(_data_layer):

    def __str__(self):
        out = []
        colnames = self._cpp_backend.colnames(self._name)
        nrows = min(6, self._cpp_backend.rows(self._name))
        locations = self.coordinates[:nrows]
        
        # prepare display
        # first column display geometrical information
        geo = [f"({coord[0]:.6f}, {coord[1]:.6f})" for coord in locations]
        geo_column = ["", "<POINT>"] + geo
        out.append(geo_column)
        # display data column
        for colname in colnames:
            v = _gf_cpp_access(self, range(nrows), colname)
            dtype = self._cpp_backend.dtype(self._name, colname)

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

        # Header
        header = "".join(col.rjust(w) for col, w in zip([c[0] for c in out], col_widths))
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

    def plot(self, ax = None, aspect = 1, boundary_nodes = None, **kwargs):
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        import math

        colnames = self._cpp_backend.colnames(self._name)
        ncols = len(colnames)
        locations = self.coordinates

        ## defaults
        if "cmap" not in kwargs:
            kwargs["cmap"] = sns.color_palette("mako", as_cmap = True)
        if "s" not in kwargs:
            kwargs["s"] = 1
        if "edgecolor" not in kwargs:
            kwargs["edgecolor"] = "none"
            
        # compute grid layout
        ncols_grid = int(math.ceil(math.sqrt(ncols)))
        nrows_grid = int(math.ceil(ncols / ncols_grid))
        
        if ax is None: ## create new panel if user didn't provide one
            fig, ax = plt.subplots(nrows_grid, ncols_grid)

        # normalize ax to ndarray
        if not isinstance(ax, np.ndarray):
            ax = np.array([ax])
        else:
            ax = ax.flatten()
        
        # plot each column
        for i, colname in enumerate(colnames):
            # data scatter
            sc = ax[i].scatter(
                x = locations[:, 0],
                y = locations[:, 1],
                c = self.col(colname),
                **kwargs
            )

            ## plot domain boundary
            if boundary_nodes is not None:
                ax[i].plot(
                    np.concatenate((boundary_nodes[:, 0], [boundary_nodes[0, 0]])),
                    np.concatenate((boundary_nodes[:, 1], [boundary_nodes[0, 1]])),
                    color = "black",
                    linewidth = 1
                )
            ax[i].set_title(colname)
            ax[i].set_aspect(aspect)
            
            # set colorbar
            cbar = ax[i].figure.colorbar(sc, ax = ax[i], fraction = 0.03)

        # hide unused axes
        for j in range(i + 1, len(ax)):
            ax[j].set_visible(False)

        fig.tight_layout(pad = 2)
        return ax
    
    @property
    def coordinates(self): return self._cpp_backend.point_coordinates(self._name)

class _areal_layer(_data_layer):
    pass
