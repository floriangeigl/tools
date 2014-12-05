import datetime
import sys


def print_f(*args, **kwargs):
    print bcolors.BLUE + '[' + str(datetime.datetime.now().replace(microsecond=0).time()) + ']' + bcolors.ENDC,
    try:
        class_name = str(kwargs['class_name'])
        print bcolors.GREEN + '{' + class_name + '}' + bcolors.ENDC,
    except KeyError:
        pass

    for i in args:
        print str(i),
    print ''
    sys.stdout.flush()


def print_fm(*args, **kwargs):
    print_f(*args, class_name='Main', **kwargs)


class bcolors:
    prefix = '\33'
    ENDC = prefix + '[0m'
    gen_c = lambda x: bcolors.ENDC + bcolors.prefix + '[' + str(x) + 'm'
    HEADER = gen_c(95)
    BLUE = gen_c(94)
    GREEN = gen_c(92)
    WARNING = gen_c(93)
    FAIL = gen_c(91)
    BLACK = gen_c('0;31')


def color_string(string, type=bcolors.BLUE):
    return type + str(string) + bcolors.ENDC

