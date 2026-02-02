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

def gcv(optimizer, edf = "stochastic", mc_samples = 100, seed = None):
    args = {
        "mc_samples" : mc_samples,
        "seed" : -1 if seed == None else seed
    }
    args.update(optimizer)
    return args

def grid_search(grid):
    import numpy as np
    args = {
        "opt" : "grid",
        "grid" : np.asarray(grid)
    }
    return args
