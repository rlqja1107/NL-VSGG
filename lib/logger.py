# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys
import shutil
import torch
#import ipdb
from tensorboardX import SummaryWriter
from termcolor import colored
import torch.distributed as dist


DEBUG_PRINT_ON = True


TFBoardHandler_LEVEL = 4


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log
    
def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def debug_print(logger, info):
    if DEBUG_PRINT_ON:
        logger.info('#'*20+' '+info+' '+'#'*20)

def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    logger = logging.getLogger(name)

    for each in logger.handlers:
        logger.removeHandler(each)

    logger.setLevel(TFBoardHandler_LEVEL)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter =  _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
            )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


    if save_dir:

        tf = TFBoardHandler(TFBoardWriter(save_dir))
        tf.setLevel(TFBoardHandler_LEVEL)
        logger.addHandler(tf)

        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger



class TFBoardWriter:
    def __init__(self, log_dir):
        if log_dir and get_rank() == 0:
            tfbd_dir = os.path.join(log_dir, 'tfboard')
            if os.path.exists(tfbd_dir):
                shutil.rmtree(tfbd_dir)
            os.makedirs(tfbd_dir)

            self.tf_writer = SummaryWriter(log_dir=tfbd_dir, flush_secs=10)
            self.enable = True
        else:
            self.enable = False
            self.tf_writer = None

    def write_data(self, meter, iter):
        if isinstance(iter, str):
            model = meter[0]
            input = meter[1]

            self.tf_writer.add_graph(model, input)
        else:
            for each in meter.keys():
                val = meter[each]
                if isinstance(val, SmoothedValue):
                    val = val.avg
                self.tf_writer.add_scalar(each, val, iter)

    def close(self):
        if self.tf_writer is not None:
            self.tf_writer.close()


class TFBoardHandler(logging.Handler):
    def __init__(self, writer):
        logging.Handler.__init__(self, TFBoardHandler_LEVEL)
        self.tf_writer = writer

    def emit(self, record):
        if record.levelno <= TFBoardHandler_LEVEL:
            self.tf_writer.write_data(record.msg[0], record.msg[1])
        return

    def close(self):
        self.tf_writer.close()