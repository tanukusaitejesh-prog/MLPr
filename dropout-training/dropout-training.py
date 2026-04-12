import numpy as np

def dropout(x, p, rng=None):
    """
    Apply inverted dropout (training mode).

    Returns:
        output, dropout_pattern
    """
    x = np.asarray(x, dtype=float)

    # Edge case: no dropout
    if p == 0:
        pattern = np.ones_like(x)
        return x.copy(), pattern

    # Choose random generator
    if rng is not None:
        rand = rng.random(x.shape)
    else:
        rand = np.random.random(x.shape)

    # Keep mask: True where we keep values
    keep = rand < (1 - p)

    # Create dropout pattern: 0 or 1/(1-p)
    pattern = keep.astype(float) / (1 - p)

    # Apply dropout
    output = x * pattern

    return output, pattern