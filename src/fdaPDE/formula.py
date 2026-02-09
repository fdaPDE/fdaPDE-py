## This file is part of fdaPDE, a C++ library for physics-informed
## spatial and functional data analysis.
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

class _formula:
    def __init__(self, formula):
        # split into lhs and rhs
        lhs, rhs = map(str.strip, formula.split("~"))
        self._lhs = [v.strip() for v in lhs.split("+")]
        # collect rhs variable names
        self._rhs = [v.strip() for v in rhs.split("+")]

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs

    
