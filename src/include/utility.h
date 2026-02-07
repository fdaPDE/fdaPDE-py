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

#ifndef __FDAPDE_PY_UTILITY_H__
#define __FDAPDE_PY_UTILITY_H__

namespace fdapde {
namespace py {

struct erased_storage {
    erased_storage() = default;
    erased_storage(const erased_storage&) = delete;
    erased_storage& operator=(const erased_storage&) = delete;
    erased_storage(erased_storage&& other) noexcept :
        ptr(std::exchange(other.ptr, nullptr)), destroy(std::exchange(other.destroy, nullptr)) { }
    erased_storage& operator=(erased_storage&& other) {
        ptr = std::exchange(other.ptr, nullptr);
        destroy = std::exchange(other.destroy, nullptr);
        return *this;
    }
    ~erased_storage() {
        if (ptr && destroy) destroy(ptr);
    }
    // cast to concrete type
    template <typename T> const T& cast() const { return *static_cast<const T*>(ptr); }
    template <typename T> T& cast() { return *static_cast<T*>(ptr); }

    void* ptr = nullptr;
    void (*destroy)(void*) = nullptr;
};

}   // namespace py
}   // namespace fdapde

#endif   // __PY_MESH_H__
