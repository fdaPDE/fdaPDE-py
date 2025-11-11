<div align="center"> <h1> fdaPDE-py </h1>

<h5> The python wrapper to the fdaPDE library </br> Physics-Informed Spatial and Functional Data Analysis. </h5> </div>

## Installation
To install fdaPDE-py, first make sure to have Git and:
* A C++20 compliant compiler. Supported versions:
     * Linux: `gcc` 14.2 (or higher), `clang` 17 (or higher)
	 * macOS: `apple-clang` (the XCode version of `clang`, AppleClang 17 or higher).
* The [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) linear algebra library, version 3.4.0.
* Python 3.7, or higher.

Then
```
git clone --recurse-submodules https://github.com/fdaPDE/fdaPDE-py
```

```
cd fdaPDE-py
python -m venv .venv

# For bash/zsh/sh
source .venv/bin/activate.sh

# For tcsh/csh
source .venv/bin/activate.csh

# For fish
source .venv/bin/activate.fish
```

```
# Install the required dependencies
pip install pybind11
pip install numpy
pip install scipy
pip install matplotlib
pip install seaborn
```

Finally,
```
pip install .
```
