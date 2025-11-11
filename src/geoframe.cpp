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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "include/geoframe.h"

namespace fdapde {
namespace py {

// clang-format off

using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
  
#define geoframe_pybind_interface(Triangulation)                                                                       \
       def(pybind11::init<pybind11::object>())                                                                         \
      .def(pybind11::init<pybind11::object, std::string, std::vector<int>, std::vector<std::string>>())                \
      .def("flt64_assign"                    , &GeoFrame<Triangulation>::assign<double>                     )          \
      .def("flt32_assign"                    , &GeoFrame<Triangulation>::assign<float>                      )          \
      .def("int64_assign"                    , &GeoFrame<Triangulation>::assign<std::int64_t>               )          \
      .def("int32_assign"                    , &GeoFrame<Triangulation>::assign<std::int32_t>               )          \
      .def("str_assign"                      , &GeoFrame<Triangulation>::assign<std::string>                )          \
      .def("flt64_access"                    , &GeoFrame<Triangulation>::access<double>                     )          \
      .def("flt32_access"                    , &GeoFrame<Triangulation>::access<float>                      )          \
      .def("int64_access"                    , &GeoFrame<Triangulation>::access<std::int64_t>               )          \
      .def("int32_access"                    , &GeoFrame<Triangulation>::access<std::int32_t>               )          \
      .def("str_access"                      , &GeoFrame<Triangulation>::access<std::string>                )          \
      .def("flt64_insert"                    , &GeoFrame<Triangulation>::insert<double>                     )          \
      .def("flt32_insert"                    , &GeoFrame<Triangulation>::insert<float>                      )          \
      .def("int64_insert"                    , &GeoFrame<Triangulation>::insert<std::int64_t>               )          \
      .def("int32_insert"                    , &GeoFrame<Triangulation>::insert<std::int32_t>               )          \
      .def("str_insert"                      , &GeoFrame<Triangulation>::insert<std::string>                )          \
      .def("ltype"                           , &GeoFrame<Triangulation>::ltype                              )          \
      .def("flt64_blk_insert"                , &GeoFrame<Triangulation>::blk_insert<double>                 )          \
      .def("int64_blk_insert"                , &GeoFrame<Triangulation>::blk_insert<std::int64_t>           )          \
      .def("ltype"                           , &GeoFrame<Triangulation>::ltype                              )          \
      .def("dtype"                           , &GeoFrame<Triangulation>::dtype                              )          \
      .def("rows"                            , &GeoFrame<Triangulation>::rows                               )          \
      .def("cols"                            , &GeoFrame<Triangulation>::cols                               )          \
      .def("colnames"                        , &GeoFrame<Triangulation>::colnames                           )          \
      .def("laynames"                        , &GeoFrame<Triangulation>::laynames                           )          \
      .def("colnames_all"                    , &GeoFrame<Triangulation>::colnames_all                       )          \
      .def("bbox"                            , &GeoFrame<Triangulation>::bbox                               )          \
      .def("n_nodes"                         , &GeoFrame<Triangulation>::n_nodes                            )          \
      .def("n_cells"                         , &GeoFrame<Triangulation>::n_cells                            )          \
      /* point layer */                                                                                                \
      .def("insert_scalar_point_layer"       , &GeoFrame<Triangulation>::insert_scalar_point_layer<matrix_t>)          \
      .def("insert_scalar_point_layer_nodes" , &GeoFrame<Triangulation>::insert_scalar_point_layer<int>     )          \
      .def("point_coordinates"               , &GeoFrame<Triangulation>::point_coordinates                  )          \
      /* areal layer */                                                                                                \
      .def("insert_scalar_areal_layer"       , &GeoFrame<Triangulation>::insert_scalar_areal_layer          )          \
      .def("load_shp"                        , &GeoFrame<Triangulation>::load_shp                           )          \
      .def("areal_polygons"                  , &GeoFrame<Triangulation>::areal_polygons                     )          \
      .def("incidence_matrix"                , &GeoFrame<Triangulation>::incidence_matrix                   )
	      
using cpp_geoframe_2_2 = GeoFrame<fdapde::Triangulation<2, 2>>;
PYBIND11_MODULE(_geoframe, m) {
    using triangulation_2_2 = fdapde::Triangulation<2, 2>;
    pybind11::class_<cpp_geoframe_2_2>(m, "cpp_geoframe_2_2").geoframe_pybind_interface(triangulation_2_2);
}

// clang-format on

}   // namespace py
}   // namespace fdapde
