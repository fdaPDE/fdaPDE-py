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
#include "include/fe_function.h"

namespace fdapde {
namespace py {

using fe_function_2_2_p1 = FeFunction<2, 2, FeP<1, 1>>;
PYBIND11_MODULE(_fe_function, m) {
    pybind11::class_<fe_function_2_2_p1>(m, "cpp_fe_function_2_2_p1")
        .def(pybind11::init<pybind11::object>()) 
        .def("coeff",            &fe_function_2_2_p1::coeff            )
        .def("eval",             &fe_function_2_2_p1::eval             )
        .def("grid_eval",        &fe_function_2_2_p1::grid_eval        )
        .def("n_dofs",           &fe_function_2_2_p1::n_dofs           )
        .def("l2_squared_norm",  &fe_function_2_2_p1::l2_squared_norm  )
        .def("h1_squared_norm",  &fe_function_2_2_p1::h1_squared_norm  )
        .def("l2_norm",          &fe_function_2_2_p1::l2_norm          )
        .def("h1_norm",          &fe_function_2_2_p1::h1_norm          )
        .def("cell_integrate_on",&fe_function_2_2_p1::cell_integrate_on)
        .def("set_coeff",        &fe_function_2_2_p1::set_coeff        );
}

}  // namespace py
}  // namespace fdapde
