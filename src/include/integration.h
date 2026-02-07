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

#ifndef __FDAPDE_PY_FE_INTEGRATION_H__
#define __FDAPDE_PY_FE_INTEGRATION_H__

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h> // da rimuovere
#include <fdaPDE/finite_elements.h>

#include "geometry.h"
#include "utility.h"

namespace fdapde {
namespace py {

Eigen::Matrix<double, Dynamic, Dynamic> fe_simplex_2d_p1_quadrature(const py::Mesh& mesh) {
    using Mesh_ = fdapde::Triangulation<2, 2>;
    const Mesh_& m = mesh.cast<2, 2>();

    return simplex_quadrature_nodes(m, QS2DP2);
}

}   // namespace py
}   // namespace fdapde

#endif   // __FDAPDE_PY_FE_INTEGRATION_H__
