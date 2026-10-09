import numpy as np


def evm(reference, received):
    """
    Return the EVM of received symbols against reference symbols, both
    normalised to unit power, and the normalised received symbols.

        EVM = sqrt(mean(|r - s|^2) / mean(|s|^2))

    Parameters
    ----------
    reference : array
        Reference data symbols s
    received : array
        Received data symbols, same shape as reference

    Returns
    -------
    EVM as a fraction, and the received symbols r normalised to unit power.
    """
    if np.shape(reference) != np.shape(received):
        raise ValueError(f"reference shape {np.shape(reference)} does not match received shape {np.shape(received)}")
    reference = reference / np.sqrt(np.mean(np.abs(reference) ** 2))
    received = received / np.sqrt(np.mean(np.abs(received) ** 2))
    error = np.mean(np.abs(received - reference) ** 2) / np.mean(np.abs(reference) ** 2)
    return np.sqrt(error), received
