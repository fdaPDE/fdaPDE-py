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

#ifndef __PY_UTILITY_H__
#define __PY_UTILITY_H__

namespace fdapde {
namespace py {

template <typename T> T& get_obj_as(const pybind11::object& py_obj, const std::string& attr) {
    pybind11::object ptr = py_obj.attr(pybind11::str(attr));
    return *pybind11::cast<T*>(ptr);
}

}   // namespace py
}   // namespace fdapde

#endif   // __PY_UTILITY_H__
