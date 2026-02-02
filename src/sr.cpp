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
#include "include/sr.h"

namespace fdapde {
namespace py {

// clang-format off
  
#define fe_ls_elliptic_pybind_interface(LocalDim, EmbedDim, Model)                                                     \
     def("fit"    , &fe_ls_elliptic<LocalDim, EmbedDim, Model>::fit    )                                               \
    .def("fit_gcv", &fe_ls_elliptic<LocalDim, EmbedDim, Model>::fit_gcv)                                               \
    .def("f"      , &fe_ls_elliptic<LocalDim, EmbedDim, Model>::f      )                                               \
    .def("beta"   , &fe_ls_elliptic<LocalDim, EmbedDim, Model>::beta   )                                               \
    .def("fitted" , &fe_ls_elliptic<LocalDim, EmbedDim, Model>::fitted )

// spatial regression
using cpp_sr_2_2 = sr_elliptic<2, 2>;
PYBIND11_MODULE(_sr, m) {
    pybind11::class_<fe_ls_elliptic<2, 2, fdapde::SRPDE<internals::fe_ls_elliptic>>>(m, "cpp_fe_ls_elliptic")
      .fe_ls_elliptic_pybind_interface(2, 2, fdapde::SRPDE<internals::fe_ls_elliptic>);
    pybind11::class_<cpp_sr_2_2, fe_ls_elliptic<2, 2, fdapde::SRPDE<internals::fe_ls_elliptic>>>(m, "cpp_sr_2_2")
      .def(pybind11::init<std::string, pybind11::object, std::optional<pybind11::dict>>());
}
  
// clang-format on

}   // namespace py
}   // namespace fdapde
