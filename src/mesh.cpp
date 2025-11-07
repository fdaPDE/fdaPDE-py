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

#include <pybind11/pybind11.h>
#include "include/mesh.h"

namespace fdapde {
namespace py {

#define triangulation_pybind_interface(LocalDim, EmbedDim)                                                             \
       def("nodes"                 , &TriangulationBase<LocalDim, EmbedDim>::nodes                 )                   \
      .def("cells"                 , &TriangulationBase<LocalDim, EmbedDim>::cells                 )                   \
      .def("boundary_nodes"        , &TriangulationBase<LocalDim, EmbedDim>::boundary_nodes        )                   \
      .def("n_nodes"               , &TriangulationBase<LocalDim, EmbedDim>::n_nodes               )                   \
      .def("n_cells"               , &TriangulationBase<LocalDim, EmbedDim>::n_cells               )                   \
      .def("n_boundary_nodes"      , &TriangulationBase<LocalDim, EmbedDim>::n_boundary_nodes      )                   \
      .def("bbox"                  , &TriangulationBase<LocalDim, EmbedDim>::bbox                  )                   \
      .def("measure"               , &TriangulationBase<LocalDim, EmbedDim>::measure               )                   \
      .def("marked_measure"        , &TriangulationBase<LocalDim, EmbedDim>::marked_measure        )                   \
      .def("sample"                , &TriangulationBase<LocalDim, EmbedDim>::sample                )                   \
      .def("mark_cells"            , &TriangulationBase<LocalDim, EmbedDim>::mark_cells            )                   \
      .def("cells_markers"         , &TriangulationBase<LocalDim, EmbedDim>::cells_markers         )                   \
      .def("clear_cells_markers"   , &TriangulationBase<LocalDim, EmbedDim>::clear_cells_markers   )                   \
      .def("filter_cells_by_marker", &TriangulationBase<LocalDim, EmbedDim>::filter_cells_by_marker)                   \
      .def("cell_coords"           , &TriangulationBase<LocalDim, EmbedDim>::cell_coords           )                   \
      .def("cell_measure"          , &TriangulationBase<LocalDim, EmbedDim>::cell_measure          )                   \
      .def("cell_bbox"             , &TriangulationBase<LocalDim, EmbedDim>::cell_bbox             )                   \
      .def("cell_barycenter"       , &TriangulationBase<LocalDim, EmbedDim>::cell_barycenter       )                   \
      .def("cell_circumcenter"     , &TriangulationBase<LocalDim, EmbedDim>::cell_circumcenter     )                   \
      .def("cell_diameter"         , &TriangulationBase<LocalDim, EmbedDim>::cell_diameter         )                   \
      .def("quadrature_nodes"      , &TriangulationBase<LocalDim, EmbedDim>::quadrature_nodes      )

using cpp_triangulation_2_2 = Triangulation<2, 2>;
PYBIND11_MODULE(_mesh, m) {
    pybind11::class_<TriangulationBase<2, 2>>(m, "cpp_triangulation_base").triangulation_pybind_interface(2, 2);
    pybind11::class_<cpp_triangulation_2_2, TriangulationBase<2, 2>>(m, "cpp_triangulation_2_2")
      .def(pybind11::init<pybind11::dict>()) 
      .def("neighbors"                , &Triangulation<2, 2>::neighbors                )
      .def("edges"                    , &Triangulation<2, 2>::edges                    )
      .def("n_edges"                  , &Triangulation<2, 2>::n_edges                  )
      .def("n_boundary_edges"         , &Triangulation<2, 2>::n_boundary_edges         )
      .def("boundary_edges"           , &Triangulation<2, 2>::boundary_edges           )
      .def("mark_boundary"            , &Triangulation<2, 2>::mark_boundary            )
      .def("edges_markers"            , &Triangulation<2, 2>::edges_markers            )
      .def("clear_boundary_markers"   , &Triangulation<2, 2>::clear_boundary_markers   )
      .def("filter_boundary_by_marker", &Triangulation<2, 2>::filter_boundary_by_marker)
      .def("edge_coords"              , &Triangulation<2, 2>::edge_coords              )
      .def("locate"                   , &Triangulation<2, 2>::locate                   );
}

} // namespace py
} // namespace fdapde
