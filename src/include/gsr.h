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

#ifndef __PY_GSR_H__
#define __PY_GSR_H__

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "fe_ls_elliptic.h"

namespace fdapde {
namespace py {

template <int LocalDim, int EmbedDim>
class gsr_elliptic : public fe_ls_elliptic<LocalDim, EmbedDim, fdapde::GSRPDE<internals::fe_ls_elliptic>> {
    using Base = fe_ls_elliptic<LocalDim, EmbedDim, fdapde::GSRPDE<internals::fe_ls_elliptic>>;
   public:
    gsr_elliptic() noexcept = default;
    gsr_elliptic(
      const std::string& formula, const pybind11::object& geoframe, const std::string& family,
      const std::optional<pybind11::dict>& penalty) :
        Base(formula, geoframe, penalty) {
        if (family == "bernoulli"  ) { this->model_.set_family(bernoulli_distribution());   }
        if (family == "poisson"    ) { this->model_.set_family(poisson_distribution());     }
        if (family == "exponential") { this->model_.set_family(exponential_distribution()); }
        if (family == "gamma"      ) { this->model_.set_family(gamma_distribution());       }
    }
};

}   // namespace py
}   // namespace fdapde

#endif   // __PY_GSR_H__
