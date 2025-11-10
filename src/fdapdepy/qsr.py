from ._qsr import cpp_qsr_2_2

def qsr(formula, data, level, penalty = None):
    return _qsr(formula, data, level, penalty)

def fe_elliptic(K = None, b = None, c = None, u = None):
    return {"K" : K, "b": b, "c": c, "u": u}

class _qsr:
    def __init__(self, formula, data, level, penalty):
        import numpy as np
        if penalty == None:
            self._cpp_backend = cpp_qsr_2_2(formula, data, level, None)
        else:
            mesh = data._domain._cpp_backend
            quad_nodes = mesh.quadrature_nodes()
            n_quad_nodes = quad_nodes.shape[0]
            embed_dim = data._domain.embed_dim()
            # expand PDE parameters
            params = {}
            fields = {
                "K": embed_dim * embed_dim,
                "b": embed_dim,
                "c": 1,
                "u": 1,
            }
            for name, size in fields.items():
                value = penalty.get(name)
                if value is None:
                    params[name] = np.zeros((n_quad_nodes, size))
                else:
                    params[name] = np.tile(np.asarray(value).flatten(), (n_quad_nodes, 1))
                    
            self._cpp_backend = cpp_qsr_2_2(formula, data, level, params)

    def fit(self, lambda_):
        self._cpp_backend.fit(lambda_)

    @property
    def f(self):
        return self._cpp_backend.f()

    @property
    def beta(self):
        return self._cpp_backend.beta()
    
    @property
    def fitted(self):
        return self._cpp_backend.fitted()
