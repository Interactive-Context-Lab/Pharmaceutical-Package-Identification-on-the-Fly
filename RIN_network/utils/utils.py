from __future__ import print_function
import os
import os.path as osp
from scipy import io
import datetime
import time

def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'
    return datetime.datetime.today().strftime(fmt)

def save_mat(ndarray, path):
    """Save a numpy ndarray as .mat file."""
    io.savemat(path, dict(ndarray=ndarray))

def may_make_dir(path):
    """
    Args:
      path: a dir, or result of `osp.dirname(osp.abspath(file_path))`
    Note:
      `osp.exists('')` returns `False`, while `osp.exists('.')` returns `True`!
    """
    # This clause has mistakes:
    # if path is None or '':

    if path in [None, '']:
        return
    if not osp.exists(path):
        os.makedirs(path)


class AverageMeter(object):
    """Modified from Tong Xiao's open-reid.
    Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / (self.count + 1e-20)


class RunningAverageMeter(object):
    """Computes and stores the running average and current value"""

    def __init__(self, hist=0.99):
        self.val = None
        self.avg = None
        self.hist = hist

    def reset(self):
        self.val = None
        self.avg = None

    def update(self, val):
        if self.avg is None:
            self.avg = val
        else:
            self.avg = self.avg * self.hist + val * (1 - self.hist)
        self.val = val


class RecentAverageMeter(object):
    """Stores and computes the average of recent values."""

    def __init__(self, hist_size=100):
        self.hist_size = hist_size
        self.fifo = []
        self.val = 0

    def reset(self):
        self.fifo = []
        self.val = 0

    def update(self, val):
        self.val = val
        self.fifo.append(val)
        if len(self.fifo) > self.hist_size:
            del self.fifo[0]

    @property
    def avg(self):
        assert len(self.fifo) > 0
        return float(sum(self.fifo)) / len(self.fifo)

class ReDirectSTD(object):
    """Modified from Tong Xiao's `Logger` in open-reid.
    This class overwrites sys.stdout or sys.stderr, so that console logs can
    also be written to file.
    Args:
      fpath: file path
      console: one of ['stdout', 'stderr']
      immediately_visible: If `False`, the file is opened only once and closed
        after exiting. In this case, the message written to file may not be
        immediately visible (Because the file handle is occupied by the
        program?). If `True`, each writing operation of the console will
        open, write to, and close the file. If your program has tons of writing
        operations, the cost of opening and closing file may be obvious. (?)
    Usage example:
      `ReDirectSTD('stdout.txt', 'stdout', False)`
      `ReDirectSTD('stderr.txt', 'stderr', False)`
    NOTE: File will be deleted if already existing. Log dir and file is created
      lazily -- if no message is written, the dir and file will not be created.
    """

    def __init__(self, fpath=None, console='stdout', immediately_visible=False):
        import sys
        import os
        import os.path as osp

        assert console in ['stdout', 'stderr']
        self.console = sys.stdout if console == 'stdout' else sys.stderr
        self.file = fpath
        self.f = None
        self.immediately_visible = immediately_visible
        if fpath is not None:
            # Remove existing log file.
            if osp.exists(fpath):
                os.remove(fpath)

        # Overwrite
        if console == 'stdout':
            sys.stdout = self
        else:
            sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            may_make_dir(os.path.dirname(osp.abspath(self.file)))
            if self.immediately_visible:
                with open(self.file, 'a') as f:
                    f.write(msg)
            else:
                if self.f is None:
                    self.f = open(self.file, 'w')
                self.f.write(msg)

    def flush(self):
        self.console.flush()
        if self.f is not None:
            self.f.flush()
            import os
            os.fsync(self.f.fileno())

    def close(self):
        self.console.close()
        if self.f is not None:
            self.f.close()