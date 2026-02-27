import numpy as np

def _fft_radial_grid(shape):
    m, n = shape
    fy = np.fft.fftfreq(m)
    fx = np.fft.fftfreq(n)
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX ** 2 + FY ** 2)
    return R

def lowpass_fft(image: np.ndarray, sigma_norm: float) -> np.ndarray:
    """Apply a Gaussian low-pass filter in the frequency domain.

    sigma_norm: Gaussian sigma in normalized frequency units (cycles per pixel), typically ~0.01-0.1
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError('lowpass_fft expects a 2D image')
    R = _fft_radial_grid(img.shape)
    H = np.exp(-(R ** 2) / (2 * (sigma_norm ** 2)))
    F = np.fft.fft2(img)
    out = np.real(np.fft.ifft2(F * H))
    return out.astype(np.float32)

def highpass_fft(image: np.ndarray, sigma_norm: float) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    R = _fft_radial_grid(img.shape)
    H = 1.0 - np.exp(-(R ** 2) / (2 * (sigma_norm ** 2)))
    F = np.fft.fft2(img)
    out = np.real(np.fft.ifft2(F * H))
    return out.astype(np.float32)

def bandpass_fft(image: np.ndarray, low_norm: float, high_norm: float) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    R = _fft_radial_grid(img.shape)
    H = ((R >= low_norm) & (R <= high_norm)).astype(float)
    F = np.fft.fft2(img)
    out = np.real(np.fft.ifft2(F * H))
    return out.astype(np.float32)


def lowpass_spatial(image: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian low-pass via spatial-domain Gaussian blur (sigma in pixels)."""
    img = np.asarray(image, dtype=np.float32)
    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(img, sigma=sigma_px).astype(np.float32)
    except Exception:
        # fallback: simple mean filter if gaussian not available
        kernel_size = max(1, int(round(sigma_px)))
        # crude box filter
        from scipy.signal import convolve2d

        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
        return convolve2d(img, kernel, mode='same', boundary='symm').astype(np.float32)


def highpass_spatial(image: np.ndarray, sigma_px: float) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    low = lowpass_spatial(img, sigma_px)
    return (img - low).astype(np.float32)


def dog_spatial(image: np.ndarray, sigma_low: float, sigma_high: float) -> np.ndarray:
    """Difference of Gaussians: blur(sigma_low) - blur(sigma_high)."""
    img = np.asarray(image, dtype=np.float32)
    low = lowpass_spatial(img, sigma_low)
    high = lowpass_spatial(img, sigma_high)
    return (low - high).astype(np.float32)

def wiener_filter(image: np.ndarray, mysize=3, noise=None):
    """Wrap scipy.signal.wiener when available, else fallback to a simple local mean filter."""
    img = np.asarray(image, dtype=np.float32)
    try:
        from scipy.signal import wiener

        return wiener(img, mysize=mysize, noise=noise).astype(np.float32)
    except Exception:
        # fallback: return a Gaussian-smoothed image as a weak denoiser
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(img, sigma=1.0).astype(np.float32)
