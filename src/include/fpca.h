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

#ifndef __FDAPDE_PY_FPCA_H__
#define __FDAPDE_PY_FPCA_H__

#include <fdaPDE/models.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
namespace nb = nanobind;

#include "geoframe.h"
#include "geometry.h"

namespace fdapde {
namespace py {

template <typename Model> struct fe_lr_elliptic {
    using matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using vector_t = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    // discretization dispatch loop
    static void initialize(Model& model, std::string colname, const py::GeoFrame& geoframe) {
        int local_dim = geoframe.local_dim();
        int embed_dim = geoframe.embed_dim();

        for (auto [ld, ed, init] : dispatch_table_) {
            if (local_dim == ld && embed_dim == ed) {
                init(model, colname, geoframe);
                return;
            }
        }
        return;
    }
   private:
    template <int local_dim, int embed_dim>
    static void init_(Model& model, std::string colname, const py::GeoFrame& geoframe) {
        using Triangulation = fdapde::Triangulation<local_dim, embed_dim>;
        using GeoFrame = fdapde::GeoFrame<Triangulation>;

        const GeoFrame& gf = geoframe.cast<local_dim, embed_dim>();
        const Triangulation& D = gf.template triangulation<0>();
        FeSpace Vh(D, P1<1>);
        TrialFunction f(Vh);
        TestFunction  v(Vh);
        auto a = integral(D)(dot(grad(f), grad(v)));
        ScalarField<local_dim, decltype([](const vector_t&) { return 0; })> u;
        auto F = integral(D)(u * v);

        model.discretize(fdapde::fe_ls_elliptic(a, F).get());
        model.analyze_data(colname, gf);
        return;
    }
    using init_fn = void (*)(Model&, std::string colname, const py::GeoFrame&);
    static constexpr std::array<std::tuple<int, int, init_fn>, 2> dispatch_table_ = {
      {{2, 2, &fe_lr_elliptic<Model>::init_<2, 2>},
       {3, 3, &fe_lr_elliptic<Model>::init_<3, 3>}}
    };
};  
  
class fPCA {
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;

    struct vtable {
        using matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
        using vector_t = Eigen::Matrix<double, Eigen::Dynamic, 1>;

        void                       (*fit          )(erased_storage&, int, const std::vector<double>&);
        const matrix_t&            (*scores       )(const erased_storage&);
        const matrix_t&            (*loadings     )(const erased_storage&);
        matrix_t                   (*fitted       )(const erased_storage&);
        const std::vector<double>& (*loadings_norm)(const erased_storage&);

        vtable() noexcept = default;
        template <typename Model> static vtable make_vtable() noexcept {
            vtable v;
            v.fit = [](erased_storage& storage, int rank, const std::vector<double>& lambda_grid) {
                Model& model = storage.cast<Model>();
                model.fit(rank, lambda_grid, ComputeRandSVD);
                return;
            };
            v.scores        = [](const erased_storage& storage) -> const matrix_t& {
	        return storage.cast<Model>().S();
	    };
            v.loadings      = [](const erased_storage& storage) -> const matrix_t& {
	        return storage.cast<Model>().F();
	    };
            v.fitted        = [](const erased_storage& storage) { return storage.cast<Model>().Fn(); };
            v.loadings_norm = [](const erased_storage& storage) -> const std::vector<double>& {
                return storage.cast<Model>().loadings_norm();
            };
            return v;
        }
    };
   public:
    fPCA() noexcept = default;
    fPCA(const std::string& colname, const py::GeoFrame& geoframe) {
        using model_t = fdapde::fPCA<internals::fe_ls_elliptic>;
        storage_.ptr = new model_t();
        storage_.destroy = [](void* p) { delete static_cast<model_t*>(p); };
        fe_lr_elliptic<model_t>::initialize(storage_.cast<model_t>(), colname, geoframe);
        vtable_ = vtable::make_vtable<model_t>();
    }
    void fit(int rank, const nb::dict& args) {
        std::vector<double> lambda_grid = nb::cast<std::vector<double>>(args["grid"]);
        vtable_.fit(storage_, rank, lambda_grid);
    }
    // observers
    const matrix_t& scores() const { return vtable_.scores(storage_); }       // scoring matrix
    const matrix_t& loadings() const { return vtable_.loadings(storage_); }   // loading matrix
    matrix_t fitted() const { return vtable_.fitted(storage_); }
    const std::vector<double>& loadings_norm() const { return vtable_.loadings_norm(storage_); }
   private:
    vtable vtable_;
    erased_storage storage_;
};

}   // namespace py
}   // namespace fdapde

#endif   // __R_FPCA_H__
