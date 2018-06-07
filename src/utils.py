import time
import logging
import os
import sys
import colorlog
import json
from datetime import datetime


def setup_logging(main_logs_dir, level=logging.INFO):
    root = logging.getLogger()
    root.setLevel(level)
    format = '%(asctime)s, %(process)-6s %(levelname)-5s %(module)s: %(message)s'

    date_format = '%H:%M:%S'
    if 'colorlog' in sys.modules and os.isatty(2):
        cformat = '%(log_color)s' + format
        f = colorlog.ColoredFormatter(cformat, date_format, log_colors={'DEBUG': 'reset',
                                                                        'INFO': 'green',
                                                                        'WARNING': 'bold_yellow',
                                                                        'ERROR': 'bold_red',
                                                                        'CRITICAL': 'bold_red'})
    else:
        f = logging.Formatter(format, date_format)

    ch = logging.StreamHandler()
    ch.setFormatter(f)
    root.addHandler(ch)

    os.makedirs(os.path.join(main_logs_dir, 'log'), exist_ok=True)
    log_file = os.path.join(main_logs_dir, 'log', datetime.utcnow().strftime('%Y_%m_%d__%H_%M_%S_%f.log'))
    fh = logging.FileHandler(log_file)
    ff = logging.Formatter(format, date_format)
    fh.setFormatter(ff)
    root.addHandler(fh)


def nested_list_len(v):
    s = 0
    if isinstance(v, list):
        for i in v:
            s += nested_list_len(i)
    else:
        s += 1
    return s


class Stats:
    def __init__(self, logger, name='Stats', verbose=True):
        self.logger = logger
        self.name = name
        self.verbose = verbose

        self.start_time = None
        self.duration = 0
        self.children = []

    def create_child_stats(self, name):
        child_stats = Stats(self.logger, name, verbose=False)
        self.children.append(child_stats)

        return child_stats

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration += time.time() - self.start_time

        if self.verbose:
            self.logger.info('%s %gs' % (self.name, self.duration))
            for child in self.children:
                self.logger.info('\t%s took %g%%' % (child.name, child.duration/self.duration*100))


def save_dict(dictionary, path):
    with open(path, 'w') as f:
        json.dump(dictionary, f, sort_keys=True, indent=4)


def load_dict(path):
    with open(path, 'r') as f:
        return json.load(f)


# TODO: Use more suited stuff for testing instead of those "if" conditions
if __name__ == '__main__':

    logging_test = True
    nested_list_test = True
    # Logging Test

    if logging_test:
        setup_logging()
        log = logging.getLogger(__name__)

        log.debug('Hello Debug')
        log.info('Hello Info')
        log.warning('Hello Warn')
        log.error('Hello Error')
        log.critical('Hello Critical')

    if nested_list_test:
        print('Nested List Test')
        l = [[1, 2, 3], [1, 2, [1, 2, [1, 2]]]]
        print('Array: ', l)
        print('Length %d' % len(l))
        print('Nested Length %d' % nested_list_len(l))


