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

#include "include/geoframe.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/eigen/dense.h>
namespace nb = nanobind;

namespace fdapde {
namespace py {

// clang-format on

void define_geoframe(nb::module_& m) {
    using dbl_matrix = Eigen::Matrix<double, Dynamic, Dynamic>;
    using int_matrix = Eigen::Matrix<int, Dynamic, Dynamic>;
  
    std::string pyclass_name = "GeoFrame";
    nb::class_<py::GeoFrame>(m, pyclass_name.c_str(), "GeoFrame object")
      .def(nb::init<py::Mesh&>())
      .def(
        nb::init<const py::GeoFrame&, const std::string&, const std::vector<int>&, const std::vector<std::string>&>())
      .def(
        "point_insert_layer",
        [](py::GeoFrame& self, const std::string& layer, const dbl_matrix& locs, const nb::dict& data) {
            return self.point_insert_layer(layer, locs, data);
        })
      .def_prop_ro("mesh", [](const py::GeoFrame& self) -> const py::Mesh& { return self.mesh(); })
      .def("ltype", [](const py::GeoFrame& self, std::string layer) { return self.ltype(layer); })
      .def(
        "dtype",
        [](const py::GeoFrame& self, std::string layer, std::string field) { return self.dtype(layer, field); })
      .def(
        "flt64_access",
        [](const py::GeoFrame& self, const std::string& layer, const std::vector<int>& rows, const std::string& col) {
            return self.access_flt64(layer, rows, col);
        })
      .def(
        "int32_access",
        [](const py::GeoFrame& self, const std::string& layer, const std::vector<int>& rows, const std::string& col) {
            return self.access_int32(layer, rows, col);
        })
      .def(
        "str_access",
        [](const py::GeoFrame& self, const std::string& layer, const std::vector<int>& rows, const std::string& col) {
            return self.access_str(layer, rows, col);
        })
      .def("rows", [](const py::GeoFrame& self, std::string layer) { return self.rows(layer); })
      .def("cols", [](const py::GeoFrame& self, std::string layer) { return self.cols(layer); })
      .def("colnames", [](const py::GeoFrame& self, std::string layer) { return self.colnames(layer); })
      .def(
        "point_coordinates", [](const py::GeoFrame& self, std::string layer) { return self.point_coordinates(layer); })
      .def(
        "load_shp",
        [](py::GeoFrame& self, std::string layer, std::string filename) {
            return self.load_shp(layer, filename);
        })
      .def(
        "areal_polygons", [](const py::GeoFrame& self, std::string layer) { return self.areal_polygons(layer); });
}
  
// clang-format on

}   // namespace py
}   // namespace fdapde
