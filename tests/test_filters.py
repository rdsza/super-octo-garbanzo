import numpy as np
from filters import lowpass_spatial, highpass_spatial, dog_spatial, wiener_filter


def make_test_image(size=128):
    # low-frequency background
    x = np.linspace(0, 2 * np.pi, size)
    X, Y = np.meshgrid(x, x)
    low = np.sin(X * 2) + np.cos(Y * 2)
    # add high-frequency noise
    rng = np.random.RandomState(0)
    noise = rng.normal(scale=0.5, size=(size, size))
    img = low + noise
    return img.astype(np.float32)


def test_lowpass_reduces_variance():
    img = make_test_image()
    orig_var = img.var()
    out = lowpass_spatial(img, sigma_px=2.0)
    assert out.shape == img.shape
    assert out.var() < orig_var


def test_highpass_reduces_low_freq():
    img = make_test_image()
    low = lowpass_spatial(img, sigma_px=2.0)
    high = highpass_spatial(img, sigma_px=2.0)
    # low should retain low-freq energy; high should have lower correlation with low
    corr = np.corrcoef(low.ravel(), high.ravel())[0, 1]
    assert abs(corr) < 0.9


def test_bandpass_basic():
    img = make_test_image()
    out = dog_spatial(img, sigma_low=1.0, sigma_high=3.0)
    assert out.shape == img.shape


def test_wiener_reduces_noise():
    img = make_test_image()
    noisy = img + np.random.RandomState(1).normal(scale=1.0, size=img.shape)
    out = wiener_filter(noisy)
    assert out.shape == noisy.shape
    # Expect reduced variance for typical denoisers
    assert out.var() <= noisy.var()
