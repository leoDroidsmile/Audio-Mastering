import numpy as np
from time import time
from scipy import signal, interpolate

from .log import debug
from . import Config
from .dsp import ms_to_lr, smooth_lowess


from .utils import to_db
from .dsp import lr_to_ms, size, unfold, batch_rms, rms, amplify, normalize


def normalize_reference(
        reference: np.ndarray,
        config: Config
) -> (np.ndarray, float):
    debug('Normalizing the REFERENCE...')

    reference, final_amplitude_coefficient = normalize(
        reference,
        config.threshold,
        config.min_value,
        normalize_clipped=False
    )

    if np.isclose(final_amplitude_coefficient, 1.):
        debug('The REFERENCE was not changed. There is no final amplitude coefficient')
    else:
        debug(f'The REFERENCE was normalized. '
              f'Final amplitude coefficient for the TARGET audio is: {to_db(final_amplitude_coefficient)}')

    return reference, final_amplitude_coefficient


def __calculate_piece_sizes(
        array: np.ndarray,
        max_piece_size: int,
        name: str,
        sample_rate: int
) -> (int, int, int):
    array_size = size(array)
    divisions = int(array_size / max_piece_size) + 1
    debug(f'The {name} will be didived into {divisions} pieces')

    piece_size = int(array_size / divisions)
    debug(f'One piece of the {name} has a length of {piece_size} samples or {piece_size / sample_rate:.2f} seconds')

    return array_size, divisions, piece_size


def get_lpis_and_match_rms(
        rmses: np.ndarray,
        average_rms: float
) -> (np.ndarray, float):
    loudest_piece_idxs = np.where(rmses >= average_rms)

    loudest_rmses = rmses[loudest_piece_idxs]
    match_rms = rms(loudest_rmses)
    debug(f'The current average RMS value in the loudest pieces is {to_db(match_rms)}')

    return loudest_piece_idxs, match_rms


def __extract_loudest_pieces(
        rmses: np.ndarray,
        average_rms: float,
        unfolded_mid: np.ndarray,
        unfolded_side: np.ndarray,
        name: str
) -> (np.ndarray, np.ndarray, float):
    debug(f'Extracting the loudest pieces of the {name} audio '
          f'with the RMS value more than average {to_db(average_rms)}...')
    loudest_piece_idxs, match_rms = get_lpis_and_match_rms(rmses, average_rms)

    mid_loudest_pieces = unfolded_mid[loudest_piece_idxs]
    side_loudest_pieces = unfolded_side[loudest_piece_idxs]

    return mid_loudest_pieces, side_loudest_pieces, match_rms


def get_average_rms(
        array: np.ndarray,
        piece_size: int,
        divisions: int,
        name: str
) -> (np.ndarray, np.ndarray, float):
    name = name.upper()
    unfolded_array = unfold(array, piece_size, divisions)

    debug(f'Calculating RMSes of the {name} pieces...')
    rmses = batch_rms(unfolded_array)
    average_rms = rms(rmses)

    return unfolded_array, rmses, average_rms


def __calculate_rms_coefficient(
        array_match_rms: float,
        reference_match_rms: float,
        epsilon: float
) -> float:
    rms_coefficient = reference_match_rms / max(epsilon, array_match_rms)
    debug(f'The RMS coefficient is: {to_db(rms_coefficient)}')
    return rms_coefficient


def get_rms_c_and_amplify_pair(
        array_main: np.ndarray,
        array_additional: np.ndarray,
        array_main_match_rms: float,
        reference_match_rms: float,
        epsilon: float,
        name: str
) -> (float, np.ndarray, np.ndarray):
    name = name.upper()
    rms_coefficient = __calculate_rms_coefficient(array_main_match_rms, reference_match_rms, epsilon)

    debug(f'Modifying the amplitudes of the {name} audio...')
    array_main = amplify(array_main, rms_coefficient)
    array_additional = amplify(array_additional, rms_coefficient)

    return rms_coefficient, array_main, array_additional


def analyze_levels(
        array: np.ndarray,
        name: str,
        config: Config
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float):
    name = name.upper()
    debug(f'Calculating mid and side channels of the {name}...')
    mid, side = lr_to_ms(array)
    del array

    array_size, divisions, piece_size = __calculate_piece_sizes(
        mid,
        config.max_piece_size,
        name,
        config.internal_sample_rate
    )

    unfolded_mid, rmses, average_rms = get_average_rms(mid, piece_size, divisions, name)
    unfolded_side = unfold(side, piece_size, divisions)

    mid_loudest_pieces, side_loudest_pieces, match_rms = __extract_loudest_pieces(
        rmses,
        average_rms,
        unfolded_mid,
        unfolded_side,
        name
    )

    return mid, side, mid_loudest_pieces, side_loudest_pieces, match_rms, divisions, piece_size



def __average_fft(
        loudest_pieces: np.ndarray,
        sample_rate: int,
        fft_size: int
) -> np.ndarray:
    *_, specs = signal.stft(
        loudest_pieces,
        sample_rate,
        window='boxcar',
        nperseg=fft_size,
        noverlap=0,
        boundary=None,
        padded=False
    )
    return np.abs(specs).mean((0, 2))


def __smooth_exponentially(
        matching_fft: np.ndarray,
        config: Config
) -> np.ndarray:
    grid_linear = config.internal_sample_rate * 0.5 * np.linspace(
        0,
        1,
        config.fft_size // 2 + 1
    )

    grid_logarithmic = config.internal_sample_rate * 0.5 * np.logspace(
        np.log10(4 / config.fft_size),
        0,
        (config.fft_size // 2) * config.lin_log_oversampling + 1
    )

    interpolator = interpolate.interp1d(grid_linear, matching_fft, 'cubic')
    matching_fft_log = interpolator(grid_logarithmic)

    matching_fft_log_filtered = smooth_lowess(
        matching_fft_log,
        config.lowess_frac,
        config.lowess_it,
        config.lowess_delta
    )

    interpolator = interpolate.interp1d(grid_logarithmic, matching_fft_log_filtered, 'cubic', fill_value='extrapolate')
    matching_fft_filtered = interpolator(grid_linear)

    matching_fft_filtered[0] = 0
    matching_fft_filtered[1] = matching_fft[1]

    return matching_fft_filtered


def get_fir(
        target_loudest_pieces: np.ndarray,
        reference_loudest_pieces: np.ndarray,
        name: str,
        config: Config
) -> np.ndarray:
    debug(f'Calculating the {name} FIR for the matching EQ...')

    target_average_fft = __average_fft(target_loudest_pieces, config.internal_sample_rate, config.fft_size)
    reference_average_fft = __average_fft(reference_loudest_pieces, config.internal_sample_rate, config.fft_size)

    np.maximum(config.min_value, target_average_fft, out=target_average_fft)
    matching_fft = reference_average_fft / target_average_fft

    matching_fft_filtered = __smooth_exponentially(matching_fft, config)

    fir = np.fft.irfft(matching_fft_filtered)
    fir = np.fft.ifftshift(fir) * signal.windows.hann(len(fir))

    return fir


def convolve(
        target_mid: np.ndarray,
        mid_fir: np.ndarray,
        target_side: np.ndarray,
        side_fir: np.ndarray
) -> (np.ndarray, np.ndarray):
    debug('Convolving the TARGET audio with calculated FIRs...')
    timer = time()
    result_mid = signal.fftconvolve(target_mid, mid_fir, 'same')
    result_side = signal.fftconvolve(target_side, side_fir, 'same')
    debug(f'The convolution is done in {time() - timer:.2f} seconds')

    debug('Converting MS to LR...')
    result = ms_to_lr(result_mid, result_side)

    return result, result_mid

