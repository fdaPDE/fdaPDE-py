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

#include <fdaPDE/finite_elements.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
namespace nb = nanobind;

#include "geoframe.h"
#include "geometry.h"

namespace fdapde {
namespace py {

struct fe_elliptic_data {
   private:
    using matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using vector_t = Eigen::Matrix<double, Eigen::Dynamic, 1>;
   public:
    explicit fe_elliptic_data(const std::optional<nb::dict>& dict) {
        if (dict) {
            K = nb::cast<matrix_t>((*dict)["K"]);
            b = nb::cast<matrix_t>((*dict)["b"]);
            c = nb::cast<vector_t>((*dict)["c"]);
            u = nb::cast<vector_t>((*dict)["u"]);
            isotropic = false;
        } else {
            isotropic = true;
        }
    }

    matrix_t K;
    matrix_t b;
    vector_t c;
    vector_t u;
    bool isotropic = false;
};

template <typename Model> struct fe_elliptic {
    // discretization dispatch loop
    static void initialize(
      Model& model, const std::string& formula, const py::GeoFrame& geoframe, const std::optional<nb::dict>& penalty) {
        int local_dim = geoframe.local_dim();
        int embed_dim = geoframe.embed_dim();

        for (auto [ld, ed, init] : dispatch_table_) {
            if (local_dim == ld && embed_dim == ed) {
                init(model, formula, geoframe, fe_elliptic_data(penalty));
                return;
            }
        }
        return;
    }
   private:
    using matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using vector_t = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    // problem setup for a diffusion-advection-reaction operator: -div[K * \nabla f] + b \cdot \nabla f + c * f
    template <int local_dim, int embed_dim>
    static void
    init_(Model& model, const std::string& formula, const py::GeoFrame& geoframe, const fe_elliptic_data& data) {
        using Triangulation = fdapde::Triangulation<local_dim, embed_dim>;
        using GeoFrame = fdapde::GeoFrame<Triangulation>;

        const GeoFrame& gf = geoframe.cast<local_dim, embed_dim>();
        const Triangulation& D = gf.template triangulation<0>();

        // setup differential problem
        FeSpace Vh(D, P1<1>);
        TrialFunction f(Vh);
        TestFunction  v(Vh);
        if (!data.isotropic) {
            // bilinear form
            FeCoeff<local_dim, local_dim, local_dim, matrix_t> K(data.K);   // FeCoeff inguardabili, dovremmo usare Map
            FeCoeff<local_dim, local_dim, 1, matrix_t> b(data.b);
            FeCoeff<local_dim, 1, 1, vector_t> c(data.c);
            auto a = integral(D)(dot(K * grad(f), grad(v)) + dot(b, grad(f)) * v + c * f * v);
            // linear form
            FeCoeff<local_dim, 1, 1, vector_t> u(data.u);
            auto F = integral(D)(u * v);

            model.discretize(fdapde::fe_ls_elliptic(a, F).get());   // questo get inguardabile
        } else {
            auto a = integral(D)(dot(grad(f), grad(v)));
            ScalarField<local_dim, decltype([](const vector_t&) { return 0; })> u;
            auto F = integral(D)(u * v);

            model.discretize(fdapde::fe_ls_elliptic(a, F).get());
        }
        model.analyze_data(formula, gf);
        return;
    }
    using init_fn = void (*)(Model&, const std::string&, const py::GeoFrame&, const fe_elliptic_data&);
    static constexpr std::array<std::tuple<int, int, init_fn>, 2> dispatch_table_ = {
      {//{1, 1, &fe_elliptic<Model>::init_<1, 1>},
       //{1, 2, &fe_elliptic<Model>::init_<1, 2>},
       {2, 2, &fe_elliptic<Model>::init_<2, 2>},
       //{2, 3, &fe_elliptic<Model>::init_<2, 3>},
       {3, 3, &fe_elliptic<Model>::init_<3, 3>}}
    };
};

}   // namespace py
}   // namespace fdapde

#endif   // __FDAPDE_PY_FE_ELLIPTIC_H__
