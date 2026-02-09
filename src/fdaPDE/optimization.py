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

from fdaPDE import cpp as _cpp
import numpy as np

__all__ = ["GridSearch", "BFGS", "GradientDescent"]


def GridSearch(grid):
    import numpy as np
    return {"opt": "grid", "grid": np.asarray(grid)}


def BFGS(max_iter = 100, tolerance = 0.01, step = 0.01):
    return {"opt": "bfgs", "max_iter": max_iter, "tolerance": tolerance, "step": step}


def GradientDescent(max_iter = 100, tolerance = 0.01, step = 0.01):
    return {"opt": "gradient_descent", "max_iter": max_iter, "tolerance": tolerance, "step": step}
