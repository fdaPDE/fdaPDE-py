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

#ifndef __PY_SR_H__
#define __PY_SR_H__

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "fe_ls_elliptic.h"

namespace fdapde {
namespace py {

template <int LocalDim, int EmbedDim>
class sr_elliptic : public fe_ls_elliptic<LocalDim, EmbedDim, fdapde::SRPDE<internals::fe_ls_elliptic>> {
    using Base = fe_ls_elliptic<LocalDim, EmbedDim, fdapde::SRPDE<internals::fe_ls_elliptic>>;
   public:
    sr_elliptic() noexcept = default;
    sr_elliptic(
      const std::string& formula, const pybind11::object& geoframe, const std::optional<pybind11::dict>& penalty) :
        Base(formula, geoframe, penalty) { }
};

}   // namespace py
}   // namespace fdapde

#endif   // __PY_SR_H__
