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

#ifndef __PY_FE_LS_ELLIPTIC_H__
#define __PY_FE_LS_ELLIPTIC_H__

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <fdaPDE/models.h>

#include "geoframe.h"
#include "utility.h"

namespace fdapde {
namespace py {

template <int LocalDim, int EmbedDim, typename Model> class fe_ls_elliptic {
    static constexpr int local_dim = LocalDim;
    static constexpr int embed_dim = EmbedDim;
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;

    using Triangulation = fdapde::Triangulation<local_dim, embed_dim>;
    using GeoFrame = fdapde::GeoFrame<Triangulation>;
   public:
    fe_ls_elliptic() noexcept = default;
    fe_ls_elliptic(
      const std::string& formula, const pybind11::object& geoframe, const std::optional<pybind11::dict>& penalty) {
        const GeoFrame& gf = get_obj_as<py::GeoFrame<Triangulation>>(geoframe, "_ptr").data();
        const Triangulation& D = gf.template triangulation<0>();
        FeSpace Vh(D, P1<1>);

        // discretize
        TrialFunction f(Vh);
        TestFunction v(Vh);
        if (penalty) {   // general elliptic operator
            const pybind11::dict& ls = *penalty;
            // bilinear form
            FeCoeff<local_dim, local_dim, local_dim, matrix_t> K(ls["K"].cast<matrix_t>());
            FeCoeff<local_dim, local_dim, 1, matrix_t> b(ls["b"].cast<matrix_t>());
            FeCoeff<local_dim, 1, 1, vector_t> c(ls["c"].cast<matrix_t>());
            auto a = integral(D)(dot(K * grad(f), grad(v)) + dot(b, grad(f)) * v + c * f * v);
            // linear form
            FeCoeff<local_dim, 1, 1, vector_t> u(ls["u"].cast<matrix_t>());
            auto F = integral(D)(u * v);

            model_.discretize(fdapde::fe_ls_elliptic(a, F).get());
        } else {   // fallback to isotropic laplacian penalty
            auto a = integral(D)(dot(grad(f), grad(v)));
            ScalarField<local_dim, decltype([](const vector_t&) { return 0; })> u;
            auto F = integral(D)(u * v);

            model_.discretize(fdapde::fe_ls_elliptic(a, F).get());
        }
        model_.analyze_data(formula, gf);
    }

    // fitting
    void fit(double lambda) { model_.fit(lambda); }
    // observers
    const vector_t& f() const { return model_.f(); }
    const vector_t& beta() const { return model_.beta(); }
    vector_t fitted() const { return model_.fitted(); }
   protected:
    Model model_;
};

}   // namespace py
}   // namespace fdapde

#endif   // __R_FE_LS_ELLIPTIC_H__
