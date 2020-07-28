from .log import Code, info, debug, debug_line, ModuleError
from . import Config, Export
from .stages import main
from .utils import get_temp_folder
from .checker import check, check_equality
from .dsp import channel_count, size
from .utils import random_file
from .log import Code, warning, info, debug, ModuleError

import soundfile as sf
import numpy as np
import os
import subprocess
import pickle
import zlib

import numpy as np

from .log import Code, info, debug, debug_line
from .dsp import size, strided_app_2d, batch_rms_2d, fade, clip
from datetime import timedelta

from .log import debug


def time_str(length, sample_rate) -> str:
    return str(timedelta(seconds=length // sample_rate))


def save(
        file: str,
        result: np.ndarray,
        sample_rate: int,
        subtype: str,
        name: str = 'result'
) -> None:
    name = name.upper()
    debug(f'Saving the {name} {sample_rate} Hz Stereo {subtype} to: \'{file}\'...')
    sf.write(file, result, sample_rate, subtype)
    debug(f'\'{file}\' is saved')


def create_preview(
        target: np.ndarray,
        result: np.ndarray,
        config: Config,
        preview_target: Export,
        preview_result: Export
) -> None:
    debug_line()
    info(Code.INFO_MAKING_PREVIEWS)

    target = clip(target, config.threshold)

    debug(f'The maximum duration of the preview is {config.preview_size / config.internal_sample_rate} seconds, '
          f'with the analysis step of {config.preview_analysis_step / config.internal_sample_rate} seconds')

    target_pieces = strided_app_2d(target, config.preview_size, config.preview_analysis_step)
    result_pieces = strided_app_2d(result, config.preview_size, config.preview_analysis_step)

    result_loudest_piece_idx = np.argmax(batch_rms_2d(result_pieces))

    target_piece = target_pieces[result_loudest_piece_idx].copy()
    result_piece = result_pieces[result_loudest_piece_idx].copy()

    del target, target_pieces, result_pieces

    debug_sample_begin = config.preview_analysis_step * int(result_loudest_piece_idx)
    debug_sample_end = debug_sample_begin + size(result_piece)
    debug(f'The best part to preview: '
          f'{time_str(debug_sample_begin, config.internal_sample_rate)} '
          f'- {time_str(debug_sample_end, config.internal_sample_rate)}')

    if size(result) != size(result_piece):
        fade_size = min(config.preview_fade_size, size(result_piece) // config.preview_fade_coefficient)
        target_piece, result_piece = fade(target_piece, fade_size), fade(result_piece, fade_size)

    if preview_target:
        save(preview_target.file, target_piece, config.internal_sample_rate, preview_target.subtype, 'target preview')

    if preview_result:
        save(preview_result.file, result_piece, config.internal_sample_rate, preview_result.subtype, 'result preview')


def load(file: str, file_type: str, temp_folder: str) -> (np.ndarray, int):
    file_type = file_type.upper()
    sound, sample_rate = None, None
    debug(f'Loading the {file_type} file: \'{file}\'...')
    try:
        sound, sample_rate = sf.read(file, always_2d=True)
    except RuntimeError as e:
        debug(e)
        if 'unknown format' in str(e):
            sound, sample_rate = __load_with_ffmpeg(file, file_type, temp_folder)
    if sound is None or sample_rate is None:
        if file_type == 'TARGET':
            raise ModuleError(Code.ERROR_TARGET_LOADING)
        else:
            raise ModuleError(Code.ERROR_REFERENCE_LOADING)
    debug(f'The {file_type} file is loaded')
    return sound, sample_rate


def __load_with_ffmpeg(file: str, file_type: str, temp_folder: str) -> (np.ndarray, int):
    sound, sample_rate = None, None
    debug(f'Trying to load \'{file}\' with ffmpeg...')
    temp_file = os.path.join(temp_folder, random_file(prefix='temp'))
    with open(os.devnull, 'w') as devnull:
        try:
            subprocess.check_call(
                [
                    'ffmpeg',
                    '-i',
                    file,
                    temp_file
                ],
                stdout=devnull,
                stderr=devnull
            )
            sound, sample_rate = sf.read(temp_file, always_2d=True)
            if file_type == 'TARGET':
                warning(Code.WARNING_TARGET_IS_LOSSY)
            else:
                info(Code.INFO_REFERENCE_IS_LOSSY)
            os.remove(temp_file)
        except FileNotFoundError:
            debug('ffmpeg is not found in the system! '
                  'Download, install and add it to PATH: https://www.ffmpeg.org/download.html')
        except subprocess.CalledProcessError:
            debug(f'ffmpeg cannot convert \'{file}\' to .wav!')
    return sound, sample_rate


DATABASE = 'database'

def process(
        target: str,
        characteristics: str,
        style : str,
        intensity : str,
        results: list,
        config: Config = Config(),
        preview_target: Export = None,
        preview_result: Export = None
):

    # debug_line()
    # info(Code.INFO_LOADING)

    if not results:
        raise RuntimeError(f'The result list is empty')

    # Get a temporary folder for converting mp3's
    temp_folder = config.temp_folder if config.temp_folder else get_temp_folder(results)

    # Load the target
    target, target_sample_rate = load(target, 'target', temp_folder)
    # Analyze the target
    target, target_sample_rate = check(target, target_sample_rate, config, 'target')

    # Load the reference
    # reference, reference_sample_rate = load(characteristics, 'reference', temp_folder)
    # Analyze the reference
    reference_sample_rate = 44100

    database = f'{DATABASE}/{style}-{intensity}.txt'

    f_reference = open(database, "rb")
    zipped = f_reference.read()
    f_reference.close()
    serialized = zlib.decompress(zipped)
    reference = pickle.loads(serialized)

    # Analyze the target and the reference together
    if not config.allow_equality:
        check_equality(target, reference)

    # Validation of the most important conditions
    if not (target_sample_rate == reference_sample_rate == config.internal_sample_rate)\
            or not (channel_count(target) == channel_count(reference) == 2)\
            or not (size(target) > config.fft_size and size(reference) > config.fft_size):
        raise ModuleError(Code.ERROR_VALIDATION)

    # Process
    result, result_no_limiter, result_no_limiter_normalized = main(
        target,
        reference,
        config,
        need_default=any(rr.use_limiter for rr in results),
        need_no_limiter=any(not rr.use_limiter and not rr.normalize for rr in results),
        need_no_limiter_normalized=any(not rr.use_limiter and rr.normalize for rr in results),
    )

    del reference
    if not (preview_target or preview_result):
        del target

    # debug_line()
    # info(Code.INFO_EXPORTING)

    # Save
    for required_result in results:
        if required_result.use_limiter:
            correct_result = result
        else:
            if required_result.normalize:
                correct_result = result_no_limiter_normalized
            else:
                correct_result = result_no_limiter
        save(required_result.file, correct_result, config.internal_sample_rate, required_result.subtype)

    # Creating a preview (if needed)
    if preview_target or preview_result:
        result = next(item for item in [result, result_no_limiter, result_no_limiter_normalized] if item is not None)
        create_preview(target, result, config, preview_target, preview_result)

    # debug_line()
    # info(Code.INFO_COMPLETED)
