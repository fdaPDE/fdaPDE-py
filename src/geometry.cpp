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

#include "include/geometry.h"
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
namespace nb = nanobind;

// clang-format off

namespace fdapde {
namespace py {

void define_mesh(nb::module_& m) {
    using dbl_matrix = Eigen::Matrix<double, Dynamic, Dynamic>;
    using int_matrix = Eigen::Matrix<int, Dynamic, Dynamic>;
  
    std::string pyclass_name = "Mesh";
    nb::class_<py::Mesh>(m, pyclass_name.c_str(), "Mesh object")
      .def(nb::init<const dbl_matrix&, const int_matrix&, const int_matrix&>())
      .def_prop_ro("n_nodes", [](const py::Mesh& self) { return self.n_nodes(); })
      .def_prop_ro("n_cells", [](const py::Mesh& self) { return self.n_cells(); })
      .def_prop_ro("n_faces", [](const py::Mesh& self) { return self.n_faces(); })
      .def_prop_ro("nodes"  , [](const py::Mesh& self) { return self.nodes();   })
      .def_prop_ro("cells"  , [](const py::Mesh& self) { return self.cells();   })
      .def_prop_ro("faces"  , [](const py::Mesh& self) { return self.faces();   })
      .def_prop_ro("bbox"   , [](const py::Mesh& self) { return self.bbox();    })
      .def_prop_ro("dim"    , [](const py::Mesh& self) { return self.dim();     })
      .def_prop_ro("measure", [](const py::Mesh& self) { return self.measure(); });
}

// clang-format on
  
} // namespace py
} // namespace fdapde
