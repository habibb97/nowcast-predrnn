import numpy as np

def compute_psnr(pred, target, data_range):
    """
    Menghitung PSNR antara prediksi dan target.

    Parameters:
    - pred: array numpy, data prediksi
    - target: array numpy, data target asli
    - data_range: float, rentang data maksimum (max - min)

    Returns:
    - psnr_value: float, nilai PSNR dalam desibel (dB)
    """
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    psnr_value = 20 * np.log10(data_range / np.sqrt(mse))
    return psnr_value
