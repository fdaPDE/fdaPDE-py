from ._geoframe import cpp_geoframe_2_2

def geoframe(domain):
    local_dim = domain.local_dim()
    embed_dim = domain.embed_dim()

    cpp_backend = None
    if (local_dim == 2 and embed_dim == 2):
        cpp_backend = cpp_geoframe_2_2(domain)
    
    return __geoframe(cpp_backend)

class __geoframe:
    def __init__(self, cpp_backend):
        self._cpp_backend = cpp_backend

    
        
    # def __str__(self):
    #     bbox = self.bbox()
        
    #     out = [
    #         "2D triangulation",
    #         f"Bounding box:   xmin: {bbox[0, 0]} ymin: {bbox[0, 1]} xmax: {bbox[1, 0]} ymax: {bbox[1, 1]}",
    #         f"Number of nodes: {self.n_nodes()}",
    #         f"Number of cells: {self.n_cells()}",
    #     ]

    #     return "\n".join(out)
