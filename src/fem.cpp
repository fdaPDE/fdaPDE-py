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
#include "include/fe_function.h"
#include "include/integration.h"
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
namespace nb = nanobind;

namespace fdapde {
namespace py {

void define_fem(nb::module_& m) {
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
  
    std::string pyclass_name = "FeFunction";
    nb::class_<py::FeFunction>(m, pyclass_name.c_str(), "Finite Element function object")
      .def(nb::init<py::Mesh&, int>())
      .def_prop_ro("coeff", [](const py::FeFunction& self) { return self.coeff(); })
      .def("eval", [](const py::FeFunction& self, const vector_t& p) { return self.eval(p); })
      .def("grid_eval", [](const py::FeFunction& self, const matrix_t& ps) { return self.grid_eval(ps); })
      .def_prop_ro("n_dofs", [](const py::FeFunction& self) { return self.n_dofs(); })
      .def("set_coeff"  , [](py::FeFunction& self, const vector_t& coeff) { return self.set_coeff(coeff);   })
      .def_prop_ro("l2_squared_norm"  , [](const py::FeFunction& self) { return self.l2_squared_norm();   })
      .def_prop_ro("h1_squared_norm"  , [](const py::FeFunction& self) { return self.h1_squared_norm();   })
      .def_prop_ro("l2_norm"  , [](const py::FeFunction& self) { return self.l2_norm();   })
      .def_prop_ro("h1_norm"  , [](const py::FeFunction& self) { return self.h1_norm();   });

    m.def("fe_simplex_2d_p1_quadrature", &fe_simplex_2d_p1_quadrature);
}

}  // namespace py
}  // namespace fdapde
