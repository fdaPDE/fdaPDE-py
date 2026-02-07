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

#include <nanobind/nanobind.h>
namespace nb = nanobind;

namespace fdapde {
namespace py {

void define_mesh(nb::module_& m);
void define_geoframe(nb::module_& m);
void define_fem(nb::module_& m);
void define_models(nb::module_& m);

NB_MODULE(cpp, m) {
    m.doc() = "fdaPDE Python interface";
    nb::module_ geometry = m.def_submodule("geometry", "Geometry module");
    define_mesh(geometry);

    define_geoframe(m);   // expose GeoFrame directly

    nb::module_ fem = m.def_submodule("fem", "Finite Element module");
    define_fem(fem);

    nb::module_ models = m.def_submodule("models", "Physics-Informed Statical Modeling module");
    define_models(models);
}

} // namespace py
} // namespace fdapde
