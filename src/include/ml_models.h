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

#ifndef __FDAPDE_PY_ML_MODELS_H__
#define __FDAPDE_PY_ML_MODELS_H__

#include <fdaPDE/models.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/eigen/dense.h>
namespace nb = nanobind;

#include "geometry.h"
#include "geoframe.h"
#include "fe_elliptic.h"

namespace fdapde {
namespace py {

template <typename Model> struct fe_ml_elliptic {
    using matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using vector_t = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    // discretization dispatch loop
    static void
    initialize(Model& model, const py::GeoFrame& geoframe, const std::optional<nb::dict>& penalty, vector_t& g_init) {
        int local_dim = geoframe.local_dim();
        int embed_dim = geoframe.embed_dim();

        for (auto [ld, ed, init] : dispatch_table_) {
            if (local_dim == ld && embed_dim == ed) {
                init(model, geoframe, fe_elliptic_data(penalty), g_init);
                return;
            }
        }
        return;
    }
   private:
    // problem setup for a diffusion-advection-reaction operator: -div[K * \nabla f] + b \cdot \nabla f + c * f
    template <int local_dim, int embed_dim>
    static void init_(Model& model, const py::GeoFrame& geoframe, const fe_elliptic_data& data, vector_t& g_init) {
        using Triangulation = fdapde::Triangulation<local_dim, embed_dim>;
        using GeoFrame = fdapde::GeoFrame<Triangulation>;

        const GeoFrame& gf = geoframe.cast<local_dim, embed_dim>();
        const Triangulation& D = gf.template triangulation<0>();

	FeSpace Vh(D, P1<1>);
	TrialFunction f(Vh);
        TestFunction  v(Vh);
	
        int n_dofs = Vh.n_dofs();
        double measure = D.measure();
        g_init.resize(n_dofs);
        for (int i = 0; i < n_dofs; ++i) { g_init[i] = std::log(1.0 / measure); }

        // discretize
        if (!data.isotropic) {
            // bilinear form
            FeCoeff<local_dim, local_dim, local_dim, matrix_t> K(data.K);
            FeCoeff<local_dim, local_dim, 1, matrix_t> b(data.b);
            FeCoeff<local_dim, 1, 1, vector_t> c(data.c);
            auto a = integral(D)(dot(K * grad(f), grad(v)) + dot(b, grad(f)) * v + c * f * v);
            // linear form
            FeCoeff<local_dim, 1, 1, vector_t> u(data.u);
            auto F = integral(D)(u * v);

            model.discretize(fdapde::fe_de_elliptic(a, F).get());
        } else {   // fallback to isotropic laplacian penalty
            auto a = integral(D)(dot(grad(f), grad(v)));
            ScalarField<local_dim, decltype([](const vector_t&) { return 0; })> u;
            auto F = integral(D)(u * v);

            model.discretize(fdapde::fe_de_elliptic(a, F).get());
        }
        model.analyze_data(gf);
        return;
    }
    using init_fn = void (*)(Model&, const py::GeoFrame&, const fe_elliptic_data&, vector_t&);
    static constexpr std::array<std::tuple<int, int, init_fn>, 2> dispatch_table_ = {
      {{2, 2, &fe_ml_elliptic<Model>::init_<2, 2>},
       {3, 3, &fe_ml_elliptic<Model>::init_<3, 3>}}
    };
};
  
struct ml_vtable {
    using matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using vector_t = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    void     (*fit        )(erased_storage&, double, const vector_t& g_init, const nb::dict&);
    vector_t (*density    )(const erased_storage&);
    vector_t (*log_density)(const erased_storage&);
    vector_t (*fitted     )(const erased_storage&);

    ml_vtable() noexcept = default;
    template <typename Model> static ml_vtable make_vtable() noexcept {
        ml_vtable v;
        v.fit = [](erased_storage& storage, double lambda, const vector_t& g_init, const nb::dict& args) {
            Model& model = storage.cast<Model>();
            // unpack optimization parameters
            int max_iter = nb::cast<int>   (args["max_iter"]);
            double tol   = nb::cast<double>(args["tolerance"]);
            double step  = nb::cast<double>(args["step"]);
            std::string opt_t = nb::cast<std::string>(args["opt"]);

            if (opt_t == "gradient_descent") {
                model.fit(lambda, g_init, GradientDescent<Dynamic> {max_iter, tol, step});
            }
            if (opt_t == "bfgs") { model.fit(lambda, g_init, BFGS<Dynamic> {max_iter, tol, step}); }
            return;
        };
        v.density     = [](const erased_storage& storage) { return storage.cast<Model>().density(); };
        v.log_density = [](const erased_storage& storage) { return storage.cast<Model>().log_density(); };
        v.fitted      = [](const erased_storage& storage) { return storage.cast<Model>().fn(); };
        return v;
    }
};

struct DEPDE {
    using matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using vector_t = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    DEPDE(const py::GeoFrame& geoframe, const std::optional<nb::dict>& penalty) {
        using model_t = fdapde::DEPDE<internals::fe_de_elliptic>;
        storage_.ptr = new model_t();
        storage_.destroy = [](void* p) { delete static_cast<model_t*>(p); };
        fe_ml_elliptic<model_t>::initialize(storage_.cast<model_t>(), geoframe, penalty, g_init_);
        vtable_ = ml_vtable::make_vtable<model_t>();
    }

    void fit(double lambda, const nb::dict& args) {
        return vtable_.fit(storage_, lambda, g_init_, args);
    }
    // observers
    vector_t density() const { return vtable_.density(storage_); }
    vector_t log_density() const { return vtable_.log_density(storage_); }
    vector_t fitted() const { return vtable_.fitted(storage_); }
   private:
    ml_vtable vtable_;
    erased_storage storage_;
    vector_t g_init_;
};
  
}   // namespace py
}   // namespace fdapde

#endif   // __FDAPDE_PY_ML_MODELS_H__
