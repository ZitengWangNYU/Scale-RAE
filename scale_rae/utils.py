import datetime
import logging
import logging.handlers
import os
import sys
import numpy as np

import requests

from scale_rae.constants import LOGDIR

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None

import torch
# TODO: move elsewhere?
IS_XLA_AVAILABLE = False
try:
    import torch_xla
    from torch_xla.distributed.spmd import XLAShardedTensor
    IS_XLA_AVAILABLE = True
except ImportError:
    pass

try:
    from decord import VideoReader, cpu
except ImportError:
    print("Please install pyav to use video processing functions.")

def inspect_tensor_sharding(t, **kwargs):

    # XLAShardedTensor is-a torch.Tensor
    def maybe_unwrap(t: torch.Tensor) -> torch.Tensor:
        return t.global_tensor if isinstance(t, XLAShardedTensor) else t

    sharding = torch_xla._XLAC._get_xla_sharding_spec(maybe_unwrap(t))
    return sharding


def process_video_with_decord(video_file, data_args):
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    avg_fps = round(vr.get_avg_fps() / data_args.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
    frame_time = [i/avg_fps for i in frame_idx]

    
    if data_args.video_max_frames > 0:
        if len(frame_idx) > data_args.video_max_frames or data_args.video_force_sample:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.video_max_frames, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    
    video = vr.get_batch(frame_idx).asnumpy()
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

    num_frames_to_sample = num_frames = len(frame_idx)
    # https://github.com/dmlc/decord/issues/208
    vr.seek(0)
    return video, video_time, frame_time, num_frames_to_sample


def process_video_with_decord_byframe(
    video_file, data_args, start_frame, end_frame, current_observation_frame=None
):
    try:
        vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        selected_frame = min(total_frame_num - 1, end_frame)
        avg_fps = round(vr.get_avg_fps() / data_args.video_fps)
        frame_idx = [i for i in range(start_frame, selected_frame, avg_fps)]

        if data_args.video_max_frames > 0:
            video_max_frames = data_args.video_max_frames
            if current_observation_frame is not None:
                video_max_frames -= 1
            if len(frame_idx) > video_max_frames or data_args.video_force_sample:
                uniform_sampled_frames = np.linspace(start_frame, selected_frame, video_max_frames, dtype=int)
                frame_idx = uniform_sampled_frames.tolist()

        if current_observation_frame:
            frame_idx.append(current_observation_frame)
        video = vr.get_batch(frame_idx).asnumpy()
        # https://github.com/dmlc/decord/issues/208
        vr.seek(0)
    except:
        raise SyntaxError("Video processing error")
    return video


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    # import torch
    # setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    # setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    # disable_torch_init()  # DO NOT ENABLE THIS: KILLS PERFORMANCE


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"
