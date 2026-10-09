import numpy as np


def evm(reference, received):
    """
    Return the EVM of equalised received symbols against reference symbols.

        EVM = sqrt(sum(|r - s|^2) / sum(|s|^2))

    Parameters
    ----------
    reference : array
        Reference data symbols s
    received : array
        Equalised received data symbols r, same shape as reference

    Returns
    -------
    EVM as a fraction.
    """
    if np.shape(reference) != np.shape(received):
        raise ValueError(f"reference shape {np.shape(reference)} does not match received shape {np.shape(received)}")
    error = np.sum(np.abs(received - reference) ** 2)
    power = np.sum(np.abs(reference) ** 2)
    return np.sqrt(error / power)
