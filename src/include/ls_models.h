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

#ifndef __FDAPDE_PY_LS_MODELS_H__
#define __FDAPDE_PY_LS_MODELS_H__

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

struct ls_vtable {
    using matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using vector_t = Eigen::Matrix<double, Eigen::Dynamic, 1>;
    using edf_cache_t = std::unordered_map<std::array<double, 1>, double, internals::std_array_hash<double, 1>>;

    void            (*fit    )(erased_storage&, double);
    nb::dict        (*fit_gcv)(erased_storage&, const nb::dict&, edf_cache_t&);
    const vector_t& (*f      )(const erased_storage&);
    const vector_t& (*beta   )(const erased_storage&);
    vector_t        (*fitted )(const erased_storage&);

    ls_vtable() noexcept = default;
    template <typename Model> static ls_vtable make_vtable() noexcept {
        ls_vtable v;
        v.fit     = [](erased_storage& storage, double lambda) { storage.cast<Model>().fit(lambda); };
        v.f       = [](const erased_storage& storage) -> const vector_t& { return storage.cast<Model>().f(); };
        v.beta    = [](const erased_storage& storage) -> const vector_t& { return storage.cast<Model>().beta(); };
        v.fitted  = [](const erased_storage& storage) { return storage.cast<Model>().fitted(); };
        v.fit_gcv = [](erased_storage& storage, const nb::dict& gcv_data, edf_cache_t& edf_cache) -> nb::dict {
            int mc_samples = nb::cast<int>(gcv_data["mc_samples"]);
            int seed = nb::cast<int>(gcv_data["seed"]);
            Model& model = storage.cast<Model>();
            auto gcv = model.gcv(edf_cache, mc_samples, seed);

            double optimum = std::numeric_limits<double>::quiet_NaN();
            std::vector<double> values;
            std::vector<double> points;
            std::string opt = nb::cast<std::string>(gcv_data["opt"]);

            if (opt == "grid") {
                // unpack optimization parameters
                std::vector<double> lambda_grid = nb::cast<std::vector<double>>(gcv_data["grid"]);
                points = lambda_grid;
                GridSearch<1> optimizer;
                optimizer.optimize(gcv, lambda_grid);
                optimum = optimizer.optimum()[0];
                values = optimizer.values();
            }
            // cache update
            edf_cache.insert(gcv.edf_cache().begin(), gcv.edf_cache().end());
            model.fit(optimum);   // fit with optimal smoothing level

            // export results
            nb::dict result;
            result["optimum"] = optimum;
            result["values"] = values;
            result["points"] = points;
            return result;
        };
        return v;
    }
};

struct SRPDE {
    using matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using vector_t = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    SRPDE(const std::string& formula, const py::GeoFrame& geoframe, const std::optional<nb::dict>& penalty) :
        edf_cache_() {
        using model_t = fdapde::SRPDE<internals::fe_ls_elliptic>;
        storage_.ptr = new model_t();
        storage_.destroy = [](void* p) { delete static_cast<model_t*>(p); };
        fe_elliptic<model_t>::initialize(storage_.cast<model_t>(), formula, geoframe, penalty);
        vtable_ = ls_vtable::make_vtable<model_t>();
    }

    void fit(double lambda) { return vtable_.fit(storage_, lambda); }
    nb::dict fit_gcv(const nb::dict& gcv_data) { return vtable_.fit_gcv(storage_, gcv_data, edf_cache_); }
    // observers
    const vector_t& f() const { return vtable_.f(storage_); }
    const vector_t& beta() const { return vtable_.beta(storage_); }
    vector_t fitted() const { return vtable_.fitted(storage_); }
   private:
    ls_vtable vtable_;
    erased_storage storage_;
    using edf_cache_t = std::unordered_map<std::array<double, 1>, double, internals::std_array_hash<double, 1>>;
    edf_cache_t edf_cache_;
};

struct GSRPDE {
    using matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using vector_t = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    GSRPDE(
      const std::string& formula, const py::GeoFrame& geoframe, const std::string& family,
      const std::optional<nb::dict>& penalty) :
        edf_cache_() {
        using model_t = fdapde::GSRPDE<internals::fe_ls_elliptic>;
        storage_.ptr = new model_t();
        storage_.destroy = [](void* p) { delete static_cast<model_t*>(p); };
        fe_elliptic<model_t>::initialize(storage_.cast<model_t>(), formula, geoframe, penalty);
        vtable_ = ls_vtable::make_vtable<model_t>();
	
        if (family == "bernoulli"  ) { storage_.cast<model_t>().set_family(bernoulli_distribution());   }
        if (family == "poisson"    ) { storage_.cast<model_t>().set_family(poisson_distribution());     }
        if (family == "exponential") { storage_.cast<model_t>().set_family(exponential_distribution()); }
        if (family == "gamma"      ) { storage_.cast<model_t>().set_family(gamma_distribution());       }
    }

    void fit(double lambda) { return vtable_.fit(storage_, lambda); }
    nb::dict fit_gcv(const nb::dict& gcv_data) { return vtable_.fit_gcv(storage_, gcv_data, edf_cache_); }
    // observers
    const vector_t& f() const { return vtable_.f(storage_); }
    const vector_t& beta() const { return vtable_.beta(storage_); }
    vector_t fitted() const { return vtable_.fitted(storage_); }
   private:
    ls_vtable vtable_;
    erased_storage storage_;
    using edf_cache_t = std::unordered_map<std::array<double, 1>, double, internals::std_array_hash<double, 1>>;
    edf_cache_t edf_cache_;
};

struct QSRPDE {
    using matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using vector_t = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    QSRPDE(
      const std::string& formula, const py::GeoFrame& geoframe, double alpha, const std::optional<nb::dict>& penalty) :
        edf_cache_() {
        using model_t = fdapde::QSRPDE<internals::fe_ls_elliptic>;
        storage_.ptr = new model_t();
        storage_.destroy = [](void* p) { delete static_cast<model_t*>(p); };
        fe_elliptic<model_t>::initialize(storage_.cast<model_t>(), formula, geoframe, penalty);
        vtable_ = ls_vtable::make_vtable<model_t>();

	storage_.cast<model_t>().set_level(alpha);
    }

    void fit(double lambda) { return vtable_.fit(storage_, lambda); }
    nb::dict fit_gcv(const nb::dict& gcv_data) { return vtable_.fit_gcv(storage_, gcv_data, edf_cache_); }
    // observers
    const vector_t& f() const { return vtable_.f(storage_); }
    const vector_t& beta() const { return vtable_.beta(storage_); }
    vector_t fitted() const { return vtable_.fitted(storage_); }
   private:
    ls_vtable vtable_;
    erased_storage storage_;
    using edf_cache_t = std::unordered_map<std::array<double, 1>, double, internals::std_array_hash<double, 1>>;
    edf_cache_t edf_cache_;
};
  
}   // namespace py
}   // namespace fdapde

#endif   // __FDAPDE_PY_LS_MODELS_H__
