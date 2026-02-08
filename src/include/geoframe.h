// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __PY_GEOFRAME_H__
#define __PY_GEOFRAME_H__

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/eigen/dense.h> // da rimuovere
#include <fdaPDE/geoframe.h>
namespace nb = nanobind;

#include "geometry.h"
#include "utility.h"

namespace fdapde {
namespace py {

class GeoFrame {
    using int_t = int;
    using str_t = std::string;
    using dbl_t = double;
    using dbl_matrix = Eigen::Matrix<double, Dynamic, Dynamic>;
    using int_matrix = Eigen::Matrix<int, Dynamic, Dynamic>;
  
    struct vtable {
        void (*point_insert_layer)(erased_storage&, const std::string&, const Eigen::Matrix<double, Dynamic, Dynamic>&);
        void (*point_load_vec_dbl64)(erased_storage&, std::string, std::string, const std::vector<double>&);
        void (*point_load_vec_int32)(erased_storage&, std::string, std::string, const std::vector<int>&);
        void (*point_load_vec_str  )(erased_storage&, std::string, std::string, const std::vector<std::string>&);
        dbl_matrix (*point_coordinates)(const erased_storage&, std::string);
        int  (*layer_category)(const erased_storage&, std::string);
        int  (*dtype)(const erased_storage&, std::string, std::string);
        std::vector<std::string> (*colnames)(const erased_storage&, std::string);
        int  (*rows)(const erased_storage&, std::string);
        std::vector<double> (*access_dbl64)(const erased_storage&, std::string, const std::vector<int>&, std::string);
        std::vector<int> (*access_int32)(const erased_storage&, std::string, const std::vector<int>&, std::string);
        std::vector<std::string> (*access_str)(
          const erased_storage&, std::string, const std::vector<int>&, std::string);
        void (*load_shp)(erased_storage&, std::string, std::string);
        std::vector<nb::dict> (*areal_polygons)(const erased_storage&, std::string);
        void (*blk_insert_int32)(erased_storage&, std::string, std::string, const int_matrix&);
        void (*blk_insert_flt64)(erased_storage&, std::string, std::string, const dbl_matrix&);
      
        template <typename T, typename GeoFrame>
        static std::vector<T>
        access(const erased_storage& s, const std::string& l, const std::vector<int>& r, std::string c) {
            const GeoFrame& gf = s.cast<GeoFrame>();
            fdapde::ltype ltype = gf[l].category()[0];
            auto access_ = [&]<typename GeoInfo>(GeoInfo, std::vector<T>& buff) {
                auto row_filter = geo_cast<GeoInfo>(gf[l]).select(r.begin(), r.end());
                buff.reserve(r.size());
                for (int i = 0, n = r.size(); i < n; ++i) { buff.emplace_back(row_filter.template col<T>(c)(i, 0)); }
                return buff;
            };
            std::vector<T> buff;
            if (ltype == ltype::point) { buff = access_(POINT {}, buff); }
            if (ltype == ltype::areal) { buff = access_(POLYGON {}, buff); }
            return buff;
        }

        vtable() noexcept = default;
        template <typename GeoFrame_> static vtable make_vtable() noexcept {
            vtable v;
            v.point_insert_layer = [](erased_storage& s, const std::string& l, const dbl_matrix& locs) {
                s.cast<GeoFrame_>().template insert_scalar_layer<POINT>(l, locs);
            };
            v.point_load_vec_dbl64 =
              [](erased_storage& s, std::string l, std::string f, const std::vector<double>& d) {
                  geo_cast<POINT>(s.cast<GeoFrame_>()[l]).load_vec(f, d);// questa operazione non dipende dal tipo geometrico, perchè devo fare questo cast??
              };
            v.point_load_vec_int32 =
              [](erased_storage& s, std::string l, std::string f, const std::vector<int>& d) {
                  geo_cast<POINT>(s.cast<GeoFrame_>()[l]).load_vec(f, d);// questa operazione non dipende dal tipo geometrico, perchè devo fare questo cast??
              };
            v.point_load_vec_str =
              [](erased_storage& s, std::string l, std::string f, const std::vector<std::string>& d) {
                  geo_cast<POINT>(s.cast<GeoFrame_>()[l]).load_vec(f, d);// questa operazione non dipende dal tipo geometrico, perchè devo fare questo cast??
              };
            v.layer_category = [](const erased_storage& s, std::string l) {
                return static_cast<int>(s.cast<GeoFrame_>()[l].category()[0]);
            };
            v.dtype = [](const erased_storage& s, std::string l, std::string c) {
                const GeoFrame_& gf = s.cast<GeoFrame_>();
                fdapde::ltype ltype = gf[l].category()[0];
                int dtype_ = 0;
                if (ltype == ltype::areal) {
                    dtype_ = int(geo_cast<POLYGON>(gf[l]).data().field_descriptor(c).type_id()); // questa operazione non dipende dal tipo geometrico, perchè devo fare questo cast??
                }
                if (ltype == ltype::point) {
                    dtype_ = int(geo_cast<POINT  >(gf[l]).data().field_descriptor(c).type_id());
                }
                return dtype_;
            };
            v.point_coordinates = [](const erased_storage& s, std::string l) -> dbl_matrix {
                return geo_index_cast<0, POINT>(s.cast<GeoFrame_>()[l]).coordinates();
            };
            v.colnames = [](const erased_storage& s, std::string l) {
                const GeoFrame_& gf = s.cast<GeoFrame_>();
                fdapde::ltype ltype = gf[l].category()[0];
                std::vector<std::string> cols_;
                if (ltype == ltype::areal) { cols_ = geo_cast<POLYGON>(gf[l]).data().colnames(); } // questa operazione non dipende dal tipo geometrico, perchè devo fare questo cast??
                if (ltype == ltype::point) { cols_ = geo_cast<POINT>  (gf[l]).data().colnames(); }
                return cols_;
            };
            v.rows = [](const erased_storage& s, std::string l) {
                const GeoFrame_& gf = s.cast<GeoFrame_>();
                fdapde::ltype ltype = gf[l].category()[0];
                int rows = 0;
                if (ltype == ltype::areal) { rows = geo_cast<POLYGON>(gf[l]).rows(); } // questa operazione non dipende dal tipo geometrico, perchè devo fare questo cast??
                if (ltype == ltype::point) { rows = geo_cast<POINT>  (gf[l]).rows(); }
                return rows;
            };
            v.access_dbl64 = [](const erased_storage& s, std::string l, const std::vector<int>& r, std::string c) {
                return vtable::access<double, GeoFrame_>(s, l, r, c);
            };
            v.access_int32 = [](const erased_storage& s, std::string l, const std::vector<int>& r, std::string c) {
                return vtable::access<int, GeoFrame_>(s, l, r, c);
            };
            v.access_str = [](const erased_storage& s, std::string l, const std::vector<int>& r, std::string c) {
                return vtable::access<std::string, GeoFrame_>(s, l, r, c);
            };
            v.load_shp = [](erased_storage& s, std::string l, std::string filename) {
                s.cast<GeoFrame_>().load_shp(l, filename);
            };
            v.areal_polygons = [](const erased_storage& s, std::string l) {
                const auto& layer = geo_index_cast<0, POLYGON>(s.cast<GeoFrame_>()[l]);
                const BinaryMatrix<Dynamic, Dynamic>& incidence_mtx = layer.incidence_matrix();
                const auto& triangulation = s.cast<GeoFrame_>().template triangulation<0>();

                std::vector<nb::dict> polygons;
                for (int i = 0; i < incidence_mtx.rows(); ++i) {
                    std::unordered_set<int> edge_ids;
                    for (int j = 0; j < incidence_mtx.cols(); ++j) {
                        if (incidence_mtx(i, j)) {
                            // loop over cell edges, take all edges which are not shared by any other cell
                            auto cell = triangulation.cell(j);
                            for (auto it = cell.edges_begin(); it != cell.edges_end(); ++it) {
                                int edge_id = it->id();
                                if (edge_ids.contains(edge_id)) {
                                    edge_ids.erase(edge_id);
                                } else {
                                    edge_ids.insert(edge_id);
                                }
                            }
                        }
                    }

                    // build polygon nodes - edges pair
                    Eigen::Matrix<int, Dynamic, Dynamic> edges(edge_ids.size(), 2);
                    std::vector<double> nodes_vec;
                    std::unordered_map<int, int> nodes_map;   // node renumbering map
                    int h = 0, k = 0;
                    for (int j : edge_ids) {
                        for (int n = 0; n < 2; n++) {
                            int node = triangulation.edges()(j, n);
                            if (nodes_map.find(node) == nodes_map.end()) {   // never found node
                                nodes_vec.push_back(triangulation.nodes()(node, 0));
                                nodes_vec.push_back(triangulation.nodes()(node, 1));
                                nodes_map.insert({node, k});
                                edges(h, n) = k;   // local node renumbering
                                k++;
                            } else {
                                edges(h, n) = nodes_map.at(node);
                            }
                        }
                        h++;
                    }
                    Eigen::Matrix<double, Dynamic, Dynamic> nodes =
                      Eigen::Map<Eigen::Matrix<double, Dynamic, Dynamic, Eigen::RowMajor>>(nodes_vec.data(), k, 2);
                    nb::dict polygon;
                    polygon["nodes"] = nodes;
                    polygon["edges"] = edges;
                    polygons.push_back(polygon);
                }
                return polygons;
            };
            v.blk_insert_int32 = [](erased_storage& s, std::string l, std::string c, const int_matrix& data) {
                s.cast<GeoFrame_>()[l].add_block(c, data);
            };
            v.blk_insert_flt64 = [](erased_storage& s, std::string l, std::string c, const dbl_matrix& data) {
                s.cast<GeoFrame_>()[l].add_block(c, data);
            };
            return v;
        }
    };
    vtable vtable_;
    erased_storage storage_;
    int local_dim_, embed_dim_;
    const Mesh* mesh_;

    // type-erased constructor routines
    // construct from Mesh
    template <int local_dim, int embed_dim> void* construct_from_mesh_(Mesh& mesh) {
        using Mesh_ = fdapde::Triangulation<local_dim, embed_dim>;
        using GeoFrame_ = fdapde::GeoFrame<Mesh_>;
        GeoFrame_* gf = new GeoFrame_(mesh.cast<local_dim, embed_dim>());
        storage_.ptr = gf;
        storage_.destroy = [](void* ptr) { delete static_cast<fdapde::GeoFrame<Mesh_>*>(ptr); };
        vtable_ = vtable::make_vtable<GeoFrame_>();
        return storage_.ptr;
    }
    static constexpr std::array<std::tuple<int, int, void* (GeoFrame::*)(Mesh&)>, 3> mesh_constructor_table_ = {
      {// std::make_tuple(1, 1, &Mesh::alloc_<1, 1>),
       // std::make_tuple(1, 2, &Mesh::alloc_<1, 2>),
       std::make_tuple(2, 2, &GeoFrame::construct_from_mesh_<2, 2>),
       std::make_tuple(2, 3, &GeoFrame::construct_from_mesh_<2, 3>),
       std::make_tuple(3, 3, &GeoFrame::construct_from_mesh_<3, 3>)}
    };
    // construct from GeoFrame
    template <int local_dim, int embed_dim>
    void* construct_from_geoframe_(
      const GeoFrame& geoframe, const std::string& layer_name, const std::vector<int>& rows,
      const std::vector<std::string>& cols) {
        using Mesh_ = fdapde::Triangulation<local_dim, embed_dim>;
        using GeoFrame_ = fdapde::GeoFrame<Mesh_>;
        auto& old_gf = geoframe.storage().cast<GeoFrame_>();

        GeoFrame_* new_gf = new GeoFrame_(const_cast<Mesh_&>(geoframe.mesh().cast<local_dim, embed_dim>()));
        storage_.ptr = new_gf;
        storage_.destroy = [](void* ptr) { delete static_cast<GeoFrame_*>(ptr); }; // -------
        vtable_ = vtable::make_vtable<GeoFrame_>();

        auto make_ = [&]<typename GeoInfo>(GeoInfo) {
            auto row_filter = geo_cast<GeoInfo>(old_gf[layer_name]).select(rows.begin(), rows.end());
            new_gf->template insert_scalar_layer<GeoInfo>(layer_name, row_filter, cols);
        };
        fdapde::ltype ltype = old_gf[layer_name].category()[0];
        if (ltype == ltype::point) { make_(POINT   {}); }
        if (ltype == ltype::areal) { make_(POLYGON {}); }
	return static_cast<void*>(new_gf);
    }
    static constexpr std::array<
      std::tuple<
        int, int,
        void* (GeoFrame::*)(const GeoFrame&, const std::string&, const std::vector<int>&,
                            const std::vector<std::string>&)>,
      3>
      geoframe_constructor_table_ = {
        {// std::make_tuple(1, 1, &Mesh::alloc_<1, 1>),
         // std::make_tuple(1, 2, &Mesh::alloc_<1, 2>),
         std::make_tuple(2, 2, &GeoFrame::construct_from_geoframe_<2, 2>),
         std::make_tuple(2, 3, &GeoFrame::construct_from_geoframe_<2, 3>),
         std::make_tuple(3, 3, &GeoFrame::construct_from_geoframe_<3, 3>)}
    };

    template <typename ConstructorTable, typename... Args>
    void construct_(int local_dim, int embed_dim, const ConstructorTable& table, Args&&... args) {
        local_dim_ = local_dim;
        embed_dim_ = embed_dim;
        for (auto [ld, ed, alloc] : table) {
            if (local_dim_ == ld && embed_dim_ == ed) {
                (this->*alloc)(std::forward<Args>(args)...);
                return;
            }
        }
        return;
    }
   public:
    GeoFrame() noexcept = default;
    // construct empty from mesh
    GeoFrame(py::Mesh& mesh) : mesh_(std::addressof(mesh)) {
        auto [ld, ed] = mesh.dim();
        construct_(ld, ed, mesh_constructor_table_, mesh);
    }
    // construct from layer, questa semantica non mi piace, in teoria dovrei poter essere in grado di creare un geoframe vuoto, costruire il layer subsettatto e inserire il layer subsettato
    GeoFrame(
      const py::GeoFrame& geoframe, const std::string& layer_name, const std::vector<int>& rows,
      const std::vector<std::string>& cols) : mesh_(std::addressof(geoframe.mesh())) {
        auto [ld, ed] = geoframe.mesh().dim();
        construct_(ld, ed, geoframe_constructor_table_, geoframe, layer_name, rows, cols);
    }

    // observers
    const Mesh& mesh() const { return *mesh_; }
    int local_dim() const { return local_dim_; }
    int embed_dim() const { return embed_dim_; }
    const erased_storage& storage() const { return storage_; }
    erased_storage& storage() { return storage_; }

    template <int LocalDim, int EmbedDim>
    const fdapde::GeoFrame<fdapde::Triangulation<LocalDim, EmbedDim>>& cast() const {
        return storage_.cast<fdapde::GeoFrame<fdapde::Triangulation<LocalDim, EmbedDim>>>();
    }
    template <int LocalDim, int EmbedDim> fdapde::GeoFrame<fdapde::Triangulation<LocalDim, EmbedDim>>& cast() {
        return storage_.cast<fdapde::GeoFrame<fdapde::Triangulation<LocalDim, EmbedDim>>>();
    }

    // layer insertion
    void point_insert_layer(const std::string& layer, const dbl_matrix& locs, const nb::dict& data) {
        vtable_.point_insert_layer(storage_, layer, locs);
        // copy input python data
        auto copy_ = [&]<typename T>(const std::string& field) {
            if (data.contains(field)) {
                nb::dict dct = nb::cast<nb::dict>(data[field.data()]);
                for (const auto& item : dct) {
                    std::string py_name = nb::cast<std::string>(item.first);
                    auto arr = nb::cast<nb::ndarray<T, nb::any_contig>>(item.second);

                    std::vector<T> vec;
                    vec.reserve(arr.size());

                    auto* ptr = arr.data();
                    std::copy(ptr, ptr + arr.size(), std::back_inserter(vec));

                    if constexpr (std::is_same_v<T, dbl_t>) {
                        vtable_.point_load_vec_dbl64(storage_, layer, py_name, vec);
                    }
                }
            }
        };
        copy_.template operator()<int_t>("int_data");
        copy_.template operator()<dbl_t>("dbl_data");
        // copy_.template operator()<str_t>("str_data"); ----------------- TODO, slow path
    }
    // void areal_insert_layer(
    //   const std::string& layer_name, const std::vector<int>& regions, const nb::dict& data) {
    //     auto& l = data_.template insert_scalar_layer<POLYGON>(layer_name, regions);
    //     // copy input python data
    //     auto copy_ = [&]<typename T>(const std::string& field) {
    //         if (data.contains(field)) {
    //             nb::dict dct = nb::cast<nb::dict>(data[field.data()]);
    //             for (const auto& item : dct) {
    //                 std::string name = nb::cast<std::string>(item.first);
    //                 nb::list py_data = nb::cast<nb::list>(item.second);
    //                 // convert into a cpp vector
    //                 std::vector<T> vec;
    //                 vec.reserve(py_data.size());
    //                 for (auto v : py_data) { vec.push_back(nb::cast<T>(v)); }
    //                 // load data
    //                 l.load_vec(name, vec);
    //             }
    //         }
    //     };
    //     copy_.template operator()<int_t>("int_data");
    //     copy_.template operator()<dbl_t>("dbl_data");
    //     copy_.template operator()<str_t>("str_data");
    // }
    void load_shp(const std::string& layer_name, const std::string& filename) {
        vtable_.load_shp(storage_, layer_name, filename);
    }
    // template <typename T>
    // void insert(const std::string& layer_name, const std::string& colname, const std::vector<T>& data) {
    //     data_[layer_name].add_column(colname, data);
    // }
    void blk_insert_flt64(
      const std::string& layer_name, const std::string& colname, const Eigen::Matrix<double, Dynamic, Dynamic>& data) {
        vtable_.blk_insert_flt64(storage_, layer_name, colname, data);
    }
    void blk_insert_int32(
      const std::string& layer_name, const std::string& colname, const Eigen::Matrix<int, Dynamic, Dynamic>& data) {
        vtable_.blk_insert_int32(storage_, layer_name, colname, data);
    }

    // template <typename T>
    // void assign(
    //   const std::string& layer_name, const std::vector<int>& rows, const std::string& column,
    //   const std::vector<T>& values) {
    //     fdapde::ltype ltype = data_[layer_name].category()[0];
    //     auto assign_ = [&]<typename GeoInfo>(GeoInfo, geoframe_t& gf) {
    //         auto row_filter =
    //           geo_cast<GeoInfo>(gf[layer_name]).select(rows.begin(), rows.end()).template col<T>(column);
    //         for (int i = 0, n = rows.size(); i < n; ++i) { row_filter(i, 0) = values[i]; }
    //     };
    //     if (ltype == ltype::point) { assign_(POINT   {}, data_); }
    // 	if (ltype == ltype::areal) { assign_(POLYGON {}, data_); }
    // }

    std::vector<double>
    access_flt64(const std::string& layer_name, const std::vector<int>& rows, const std::string& column) const {
        return vtable_.access_dbl64(storage_, layer_name, rows, column);
    }
    std::vector<int>
    access_int32(const std::string& layer_name, const std::vector<int>& rows, const std::string& column) const {
        return vtable_.access_int32(storage_, layer_name, rows, column);
    }
    std::vector<std::string>
    access_str  (const std::string& layer_name, const std::vector<int>& rows, const std::string& column) const {
        return vtable_.access_str  (storage_, layer_name, rows, column);
    }

    // in all these operations we must cast to a geo_type, but these operations are NOT geometric aware.... correct
    int ltype(const std::string& layer_name) const { return vtable_.layer_category(storage_, layer_name); }
    int dtype(const std::string& layer_name, const std::string& colname) const {
        return vtable_.dtype(storage_, layer_name, colname);
    }
    std::vector<std::string> colnames(const std::string& layer_name) const {
        return vtable_.colnames(storage_, layer_name);
    }
    int cols(const std::string& layer_name) const { return colnames(layer_name).size(); }
    int rows(const std::string& layer_name) const { return vtable_.rows(storage_, layer_name); }
  
    // std::vector<std::string> laynames() const { return data_.laynames(); }
  
    // areal layer
    std::vector<nb::dict> areal_polygons(const std::string& layer_name) const {
        return vtable_.areal_polygons(storage_, layer_name);
    }
    // Eigen::Matrix<int, Dynamic, Dynamic> incidence_matrix(const std::string& layer_name) const {
    //     const auto& layer = geo_index_cast<0, POLYGON>(data_[layer_name]);

    //     const BinaryMatrix<Dynamic, Dynamic>& incidence_matrix = layer.incidence_matrix();
    // 	int rows = incidence_matrix.rows();
    // 	int cols = incidence_matrix.cols();
    //     Eigen::Matrix<int, Dynamic, Dynamic> mat(rows, cols);
    //     for (int i = 0; i < rows; ++i) {
    //         for (int j = 0; j < cols; ++j) {
    //             if (incidence_matrix(i, j))
    //                 mat(i, j) = 1;
    //             else
    //                 mat(i, j) = 0;
    //         }
    //     }
    //     return mat;
    // }

    // point layer
    Eigen::Matrix<double, Dynamic, Dynamic> point_coordinates(const std::string& layer_name) const {
        return vtable_.point_coordinates(storage_, layer_name);
    }
};

}   // namespace py
}   // namespace fdapde

#endif // __PY_GEOFRAME_H__
