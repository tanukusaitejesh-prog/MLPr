import numpy as np

def clip_gradients(g, max_norm):
    # Convert input to numpy array
    g = np.asarray(g, dtype=float)
    
    # Compute L2 norm
    norm = np.linalg.norm(g)
    
    # Edge cases: zero norm or invalid max_norm
    if norm == 0 or max_norm <= 0:
        return g.copy()
    
    # No clipping needed
    if norm <= max_norm:
        return g.copy()
    
    # Scale gradients
    return g * (max_norm / norm)