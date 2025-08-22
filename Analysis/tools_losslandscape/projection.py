
import numpy as np

import torch



def nplist_to_tensor(nplist):
    """ Concatenate a list of numpy vectors into one tensor.

        Args:
            nplist: a list of numpy vectors, e.g., direction loaded from h5 file.

        Returns:
            concatnated 1D tensor
    """
    v = []
    for d in nplist:
        w = torch.tensor(d*np.float64(1.0))
        # Ignoreing the scalar values (w.dim() = 0).
        if w.dim() > 1:
            v.append(w.view(w.numel()))
        elif w.dim() == 1:
            v.append(w)
    return torch.cat(v)





def cal_angle(vec1, vec2):
    """ Calculate cosine similarities between two torch tensors or two ndarraies
        Args:
            vec1, vec2: two tensors or numpy ndarraies
    """
    if isinstance(vec1, torch.Tensor) and isinstance(vec1, torch.Tensor):
        return torch.dot(vec1, vec2)/(vec1.norm()*vec2.norm()).item()
    elif isinstance(vec1, np.ndarray) and isinstance(vec2, np.ndarray):
        return np.ndarray.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))



