from __future__ import print_function
import datetime
import sys
import warnings
import threading


def print_f(*args, **kwargs):
    try:
        if kwargs['thread_name']:
            print(color_string('[' + str(threading.current_thread().name) + ']'), end='')
    except KeyError:
        pass
    print(bcolors.BLUE + '[' + str(datetime.datetime.now().replace(microsecond=0).time()) + ']' + bcolors.ENDC, end='')
    try:
        class_name = str(kwargs['class_name'])
        print(bcolors.GREEN + '{' + class_name + '}' + bcolors.ENDC, end='')
    except KeyError:
        pass

    print(' '.join(map(str, args)))
    sys.stdout.flush()


def print_fm(*args, **kwargs):
    print_f(*args, class_name='Main', **kwargs)


class bcolors:
    prefix = '\33'
    ENDC = prefix + '[0m'
    gen_c = lambda x, ENDC=ENDC, prefix=prefix: ENDC + prefix + '[' + str(x) + 'm'
    HEADER = gen_c(95)
    BLUE = gen_c(94)
    GREEN = gen_c(92)
    WARNING = gen_c(93)
    FAIL = gen_c(91)
    BLACK = gen_c('0;31')


def color_string(string, type=bcolors.BLUE):
    return type + str(string) + bcolors.ENDC

