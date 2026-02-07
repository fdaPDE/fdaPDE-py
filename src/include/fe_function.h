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

#ifndef __FDAPDE_PY_FE_FUNCTION_H__
#define __FDAPDE_PY_FE_FUNCTION_H__

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/eigen/dense.h> // da rimuovere
#include <fdaPDE/finite_elements.h>

#include "geometry.h"
#include "utility.h"

namespace fdapde {
namespace py {

class FeFunction {
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
  
    struct vtable {
        const vector_t& (*coeff          )(const erased_storage&);
        double          (*eval           )(const erased_storage&, const vector_t&);
        vector_t        (*grid_eval      )(const erased_storage&, const matrix_t&);
        double          (*l2_squared_norm)(const erased_storage&);
        double          (*l2_norm        )(const erased_storage&);
        double          (*h1_squared_norm)(const erased_storage&);
        double          (*h1_norm        )(const erased_storage&);
        void            (*set_coeff      )(erased_storage&, const vector_t&);

        vtable() noexcept = default;
        template <typename FeFunction_> static vtable make_vtable() noexcept {
            vtable v;
            // observers
            v.coeff = [](const erased_storage& storage) -> const vector_t& {
                return storage.cast<FeFunction_>().coeff();
            };
            v.eval = [](const erased_storage& storage, const vector_t& p) -> double {
                return storage.cast<FeFunction_>()(p);
            };
            v.grid_eval = [](const erased_storage& storage, const matrix_t& ps) -> vector_t {
                vector_t res(ps.rows());
		const auto& f = storage.cast<FeFunction_>();
                for (int i = 0, n = ps.rows(); i < n; ++i) { res[i] = f(ps.row(i)); }
                return res;
            };
            v.l2_squared_norm = [](const erased_storage& storage) -> double {
                return storage.cast<FeFunction_>().l2_squared_norm();
            };
            v.l2_norm = [](const erased_storage& storage) -> double { return storage.cast<FeFunction_>().l2_norm(); };
            v.h1_squared_norm = [](const erased_storage& storage) -> double {
                return storage.cast<FeFunction_>().h1_squared_norm();
            };
            v.h1_norm = [](const erased_storage& storage) -> double { return storage.cast<FeFunction_>().h1_norm(); };
            v.set_coeff = [](erased_storage& storage, const vector_t& coeff) {
                return storage.cast<FeFunction_>().set_coeff(coeff);
            };
            return v;
        }
    };

    template <int local_dim, int embed_dim, typename FeType> void alloc_(py::Mesh& mesh) {
        using Mesh_ = fdapde::Triangulation<local_dim, embed_dim>;
        using FeSpace_ = fdapde::FeSpace<Mesh_, FeType>;
        using FeFunction_ = fdapde::FeFunction<FeSpace_>;

        Mesh_& m = mesh.cast<local_dim, embed_dim>();
        fe_space_.ptr = new FeSpace_(m, FeType {});
        fe_space_.destroy = [](void* ptr) { delete static_cast<FeSpace_*>(ptr); };

        storage_.ptr = new FeFunction_(fe_space_.cast<FeSpace_>());
        storage_.destroy = [](void* ptr) { delete static_cast<FeFunction_*>(ptr); };
        vtable_ = vtable::make_vtable<FeFunction_>();
        n_dofs_ = fe_space_.cast<FeSpace_>().n_dofs();
    }
    using alloc_fn = void (FeFunction::*)(py::Mesh&);
    static constexpr std::array<std::tuple<int, int, alloc_fn>, 2> alloc_table_ = { // primo int, dimensione della mesh, sempre 2, secondo int, ordine degli elementi P
      {// std::make_tuple(1, 1, &Mesh::alloc_<1, 1>),
       // std::make_tuple(1, 2, &Mesh::alloc_<1, 2>),
	std::make_tuple(2, 1, &FeFunction::alloc_<2, 2, FeP<1, 1>>), // ---------- qui vedi come il numero di istanziazioni esplode, bisogna rimuovere questi template che non portano a nessun vantaggio computazionale
	std::make_tuple(2, 2, &FeFunction::alloc_<2, 2, FeP<2, 1>>)}
    };  
   public:
    FeFunction() noexcept = default;
    FeFunction(py::Mesh& mesh, int fe_order) {
        int local_dim = mesh.dim()[0];
        for (auto [ld, order, alloc] : alloc_table_) {
            if (local_dim == ld && fe_order == order) {
                (this->*alloc)(mesh);
                return;
            }
        }
    }

    // observers
    const vector_t& coeff() const { return vtable_.coeff(storage_); }
    double eval(const vector_t& p) const { return vtable_.eval(storage_, p); }
    vector_t grid_eval(const matrix_t& ps) const { return vtable_.grid_eval(storage_, ps); }
    double l2_squared_norm() const { return vtable_.l2_squared_norm(storage_); }
    double l2_norm() const { return vtable_.l2_norm(storage_); }
    double h1_squared_norm() const { return vtable_.h1_squared_norm(storage_); }
    double h1_norm() const { return vtable_.h1_norm(storage_); }
    // integration
    // double cell_integrate_on(int marker) const {
    //     if (marker == BoundaryAll) {
    //         return fe_function_.integrate_on(
    //           fe_space_.triangulation().cells_begin(), fe_space_.triangulation().cells_end());
    //     } else {
    //         return fe_function_.integrate_on(
    //           fe_space_.triangulation().cells_begin(marker), fe_space_.triangulation().cells_end(marker));
    //     }
    // }
    int n_dofs() const { return n_dofs_; }
    // modifiers
    void set_coeff(const vector_t& coeff) { vtable_.set_coeff(storage_, coeff); }
   private:
    int n_dofs_;
    vtable vtable_;
    erased_storage storage_;
    erased_storage fe_space_;
};

}   // namespace py
}   // namespace fdapde

#endif   // __FDAPDE_PY_FE_FUNCTION_H__
