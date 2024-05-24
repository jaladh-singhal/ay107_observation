import numpy as np
import matplotlib.pyplot as plt


# Plotting ------------------------------------------------------------

def clim(data, prange=(1, 99)):
    return (np.percentile(data, prange[0]), np.percentile(data, prange[1]))


def plot_frame(frame, figsize=(15, 3), aspect=None, prange=(1, 99), label=None, title=None):
    plt.figure(figsize=figsize)
    plt.imshow(frame, origin='lower', clim=clim(frame, prange), aspect=aspect)
    plt.colorbar(label=label)
    if title:
        plt.title(title)


def plot_frame2(frame, figsize=(15, 3), aspect=None, prange=(1, 99), label=None, title=None):
    fig, ax = plt.subplots(figsize=figsize)
    h = ax.imshow(frame, origin='lower', clim=clim(frame, prange), aspect=aspect)
    plt.colorbar(h, label=label)
    if title:
        ax.set_title(title)
    return fig, ax


# Data transformation ---------------------------------------------------

def reduce_frame(frame, exptime, master_flat, master_bias):
    return ((frame - master_bias)/exptime) / master_flat

def rectify_frame(frame, trace_y, y_bound_upper, y_bound_lower):
    rectified_frame = []
    for xi in range(frame.shape[1]):
        rectified_frame.append(frame[trace_y[xi]-y_bound_lower : trace_y[xi]+y_bound_upper+1, xi])
    return np.array(rectified_frame).T


def correct_airmass(star_wvl, star_frame, extinction_wvl, extinction, airmass):
    airmass_extinction = extinction * airmass
    airmass_extinction_wvl_dep = np.interp(star_wvl,
                                           extinction_wvl,
                                           airmass_extinction)
    zero_airmass_flux = star_frame * (10 ** (airmass_extinction_wvl_dep / 2.5))
    return zero_airmass_flux


def extract_source_spectrum(science_data, aperture_idx_range):
    return np.sum(science_data[aperture_idx_range[0]: aperture_idx_range[1] + 1], axis=0)


def extract_bg_spectrum(science_data, bg_lower_idx, bg_upper_idx):
    return np.median(np.concatenate((
        science_data[bg_lower_idx[0]:bg_lower_idx[1] + 1],
        science_data[bg_upper_idx[0]:bg_upper_idx[1] + 1]),
        axis=0), axis=0)


def spectrum_wo_bg(source_spectrum, aperture_idx_range, bg_spectrum):
    aperture_nrows = aperture_idx_range[1] - aperture_idx_range[0] + 1
    return source_spectrum - (bg_spectrum * aperture_nrows)


def gaussian(x, h, mu, sigma):
    return h * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
