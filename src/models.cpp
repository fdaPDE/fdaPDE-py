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
#include <nanobind/stl/optional.h>
#include <nanobind/eigen/dense.h>
#include "include/ls_models.h"
#include "include/ml_models.h"
#include "include/fpca.h"

namespace fdapde {
namespace py {

void define_models(nb::module_& m) {
    using vector_t = Eigen::Matrix<double, Eigen::Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  
    nb::class_<py::SRPDE>(m, std::string("SRPDE").c_str(), "Spatial Regression model")
      .def(
        nb::init<const std::string&, const py::GeoFrame&, const std::optional<nb::dict>&>(),
        nb::arg("formula"),
        nb::arg("data"),
        nb::arg("penalty") = nb::none())
      .def("fit"    , [](py::SRPDE& self, double lambda) { return self.fit(lambda); }                 )
      .def("fit_gcv", [](py::SRPDE& self, const nb::dict& gcv_data) { return self.fit_gcv(gcv_data); })
      .def("f"      , [](const py::SRPDE& self) { return self.f(); }                                  )
      .def("beta"   , [](const py::SRPDE& self) { return self.beta(); }                               )
      .def("fitted" , [](const py::SRPDE& self) { return self.fitted(); }                             );

    nb::class_<py::GSRPDE>(m, std::string("GSRPDE").c_str(), "Generalized Spatial Regression model")
      .def(
        nb::init<const std::string&, const py::GeoFrame&, const std::string&, const std::optional<nb::dict>&>(),
        nb::arg("formula"),
        nb::arg("data"),
        nb::arg("family"),
        nb::arg("penalty") = nb::none())
      .def("fit"    , [](py::GSRPDE& self, double lambda) { return self.fit(lambda); }                 )
      .def("fit_gcv", [](py::GSRPDE& self, const nb::dict& gcv_data) { return self.fit_gcv(gcv_data); })
      .def("f"      , [](const py::GSRPDE& self) { return self.f(); }                                  )
      .def("beta"   , [](const py::GSRPDE& self) { return self.beta(); }                               )
      .def("fitted" , [](const py::GSRPDE& self) { return self.fitted(); }                             );

    nb::class_<py::QSRPDE>(m, std::string("QSRPDE").c_str(), "Quantile Spatial Regression model")
      .def(
        nb::init<const std::string&, const py::GeoFrame&, double, const std::optional<nb::dict>&>(),
        nb::arg("formula"),
        nb::arg("data"),
        nb::arg("level"),
        nb::arg("penalty") = nb::none())
      .def("fit"    , [](py::QSRPDE& self, double lambda) { return self.fit(lambda); }                 )
      .def("fit_gcv", [](py::QSRPDE& self, const nb::dict& gcv_data) { return self.fit_gcv(gcv_data); })
      .def("f"      , [](const py::QSRPDE& self) { return self.f(); }                                  )
      .def("beta"   , [](const py::QSRPDE& self) { return self.beta(); }                               )
      .def("fitted" , [](const py::QSRPDE& self) { return self.fitted(); }                             );

    nb::class_<py::DEPDE>(m, std::string("DEPDE").c_str(), "Density Estimation model")
      .def(
        nb::init<const py::GeoFrame&, const std::optional<nb::dict>&>(),
        nb::arg("data"),
        nb::arg("penalty") = nb::none())
      .def("fit"        , [](py::DEPDE& self, double lambda, const nb::dict& args) { return self.fit(lambda, args); })
      .def("density"    , [](const py::DEPDE& self) { return self.density(); }                                       )
      .def("log_density", [](const py::DEPDE& self) { return self.log_density(); }                                   )
      .def("fitted"     , [](const py::DEPDE& self) { return self.fitted(); }                                        );

    nb::class_<py::fPCA>(m, std::string("fPCA").c_str(), "functional Principal Component Analysis")
      .def(
        nb::init<const std::string&, const py::GeoFrame&, const matrix_t&>(), nb::arg("colname"), nb::arg("data"),
        nb::arg("K_data"))
      .def("fit", [](py::fPCA& self, int rank, const nb::dict& args) { return self.fit(rank, args); })
      .def("scores", [](const py::fPCA& self) { return self.scores(); })
      .def("loadings", [](const py::fPCA& self) { return self.fitted(); })
      .def("pcs", [](const py::fPCA& self) { return self.loadings(); })
      .def("pcs_norm", [](const py::fPCA& self) { return self.loadings_norm(); });
}
  
// clang-format on

}   // namespace py
}   // namespace fdapde
