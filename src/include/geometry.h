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

#ifndef __PY_MESH_H__
#define __PY_MESH_H__

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/eigen/dense.h> // da rimuovere
#include <fdaPDE/geometry.h>
namespace nb = nanobind;

#include "utility.h"

namespace fdapde {
namespace py {

class Mesh {
    using double_mtx = Eigen::Matrix<double, Dynamic, Dynamic>;
    using int_mtx    = Eigen::Matrix<int, Dynamic, Dynamic>;

    struct vtable {
        int        (*n_nodes)(const erased_storage&);
        int        (*n_cells)(const erased_storage&);
        int        (*n_faces)(const erased_storage&);
        double_mtx (*nodes  )(const erased_storage&);
        int_mtx    (*cells  )(const erased_storage&);
        int_mtx    (*faces  )(const erased_storage&);
        double_mtx (*bbox   )(const erased_storage&);
        double     (*measure)(const erased_storage&);
        double_mtx (*boundary_nodes)(const erased_storage&);

        vtable() noexcept = default;
        template <typename Mesh_> static vtable make_vtable() noexcept {
            vtable v;
	    // observers
            v.n_nodes = [](const erased_storage& storage) -> int { return storage.cast<Mesh_>().n_nodes(); };
            v.n_cells = [](const erased_storage& storage) -> int { return storage.cast<Mesh_>().n_cells(); };
            v.n_faces = [](const erased_storage& storage) -> int {
                if constexpr (Mesh_::local_dim == 2) { return storage.cast<Mesh_>().n_edges(); }
                if constexpr (Mesh_::local_dim == 3) { return storage.cast<Mesh_>().n_faces(); }
            };
	    v.bbox    = [](const erased_storage& storage) -> double_mtx { return storage.cast<Mesh_>().bbox(); };
	    v.measure = [](const erased_storage& storage) -> double { return storage.cast<Mesh_>().measure(); };
            v.nodes   = [](const erased_storage& storage) -> double_mtx { return storage.cast<Mesh_>().nodes(); };
            v.cells   = [](const erased_storage& storage) -> int_mtx { return storage.cast<Mesh_>().cells(); };
            v.faces   = [](const erased_storage& storage) -> int_mtx {
                if constexpr (Mesh_::local_dim == 2) { return storage.cast<Mesh_>().edges(); }
                if constexpr (Mesh_::local_dim == 3) { return storage.cast<Mesh_>().faces(); }
            };
            return v;
        }
    };
    vtable vtable_;
    erased_storage storage_;
    int local_dim_, embed_dim_;

    template <int local_dim, int embed_dim>
    void alloc_(const double_mtx& nodes, const int_mtx& cells, const int_mtx& boundary) {
        using Mesh_ = fdapde::Triangulation<local_dim, embed_dim>;
        storage_.ptr = new Mesh_(nodes, cells, boundary);
        storage_.destroy = [](void* ptr) { delete static_cast<Mesh_*>(ptr); };
        vtable_ = vtable::make_vtable<Mesh_>();
    }
    using alloc_fn = void (Mesh::*)(const double_mtx&, const int_mtx&, const int_mtx&);
    static constexpr std::array<std::tuple<int, int, alloc_fn>, 3> alloc_table_ = {
      {// std::make_tuple(1, 1, &Mesh::alloc_<1, 1>),
       // std::make_tuple(1, 2, &Mesh::alloc_<1, 2>),
       std::make_tuple(2, 2, &Mesh::alloc_<2, 2>),
       std::make_tuple(2, 3, &Mesh::alloc_<2, 3>),
       std::make_tuple(3, 3, &Mesh::alloc_<3, 3>)}
    };
   public:
    Mesh() noexcept = default;
    Mesh(const double_mtx& nodes, const int_mtx& cells, const int_mtx& boundary) noexcept {
        // infer mesh dimensions
        local_dim_ = nodes.cols();
        embed_dim_ = cells.cols() - 1;
	// perform allocation
        for (auto [ld, ed, alloc] : alloc_table_) {
            if (local_dim_ == ld && embed_dim_ == ed) {
                (this->*alloc)(nodes, cells, boundary);
                return;
            }
        }
	return;
    }
    ~Mesh() = default;
    // observers
    std::array<int, 2> dim() const { return std::array<int, 2> {local_dim_, embed_dim_}; }
    int        n_cells() const { return vtable_.n_cells(storage_); }
    int        n_nodes() const { return vtable_.n_nodes(storage_); }
    int        n_faces() const { return vtable_.n_faces(storage_); }
    double_mtx nodes()   const { return vtable_.nodes(storage_);   }   // NB: qui dovremmo ritornare references
    int_mtx    cells()   const { return vtable_.cells(storage_);   }
    int_mtx    faces()   const { return vtable_.faces(storage_);   }
    double_mtx bbox()    const { return vtable_.bbox(storage_);    }
    double     measure() const { return vtable_.measure(storage_); }

    template <int LocalDim, int EmbedDim> const fdapde::Triangulation<LocalDim, EmbedDim>& cast() const {
        return storage_.cast<fdapde::Triangulation<LocalDim, EmbedDim>>();
    }
    template <int LocalDim, int EmbedDim> fdapde::Triangulation<LocalDim, EmbedDim>& cast() {
        return storage_.cast<fdapde::Triangulation<LocalDim, EmbedDim>>();
    }

    // void mark_cells(int marker, const std::vector<bool>& mask) {
    //     vtable_.mark_cells(marker, [&](const auto& cell) { return mask[cell.id()]; });
    // }

    //  // random sample in triangulation
    //  double_mtx sample(int n_samples, int seed = random_seed) const { return triangulation_.sample(n_samples, seed);
    //  }
    //  // cell marker handling
    //  void clear_cells_markers() { triangulation_.clear_cell_markers(); }
    //  const std::vector<int>& cells_markers() const { return triangulation_.cells_markers(); }
    //  void mark_cells(int marker, const std::vector<bool>& mask) {
    //      triangulation_.mark_cells(marker, [&](const auto& cell) { return mask[cell.id()]; });
    //  }
    //  // all indices of cells having marker = m
    //  std::vector<int> filter_cells_by_marker(int m) const {
    //      std::vector<int> cell_ids;
    //      if (!triangulation_.cells_markers().empty()) {
    //          for (int i = 0, n = triangulation_.n_cells(); i < n; ++i) {
    //              if (triangulation_.cells_markers()[i] == m) { cell_ids.push_back(i); }
    //          }
    //      }
    //      return cell_ids;
    //  }
    //  // cell properties
    //  const auto& cell_coords(int cell_id) { return triangulation_.cell(cell_id).nodes(); }
    //  double cell_measure(int cell_id) const { return triangulation_.cell(cell_id).measure(); }
    //  double_mtx cell_bbox(int cell_id) {
    //      auto [ll, ur] = triangulation_.cell(cell_id).bounding_box();
    // 	double_mtx bbox(2, embed_dim);
    // 	bbox.row(0) = ll;
    // 	bbox.row(1) = ur;
    //      return bbox;
    //  }
    //  node_t cell_barycenter(int cell_id)   const { return triangulation_.cell(cell_id).barycenter(); }
    //  node_t cell_circumcenter(int cell_id) const { return triangulation_.cell(cell_id).circumcenter(); }
    //  double cell_diameter(int cell_id) const { return triangulation_.cell(cell_id).diameter(); }
};
  
}   // namespace py
}   // namespace fdapde

#endif   // __PY_MESH_H__
