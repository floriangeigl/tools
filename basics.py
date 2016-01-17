import os
import shutil
import traceback


def create_folder_structure(filename):
    d = os.path.dirname(filename)
    if not os.path.exists(d):
        os.makedirs(d)


def delete_folder(filename, force=True):
    d = os.path.dirname(filename)
    if os.path.exists(d):
        shutil.rmtree(d)


def delete_file(filename):
    if os.path.isfile(filename):
        os.remove(filename)


def find_files(base_dir, file_ending):
    res = list()
    for root, dirs, files in os.walk(base_dir):
        if not root.endswith('/'):
            root += '/'
        res.extend([root + i for i in filter(lambda x: x.endswith(file_ending), files)])
    return sorted(res)


def try_catch_traceback_wrapper(func):
    def inner(*args, **kwargs):
        try:
            ret_val = func(*args, **kwargs)
        except Exception as e:
            print(traceback.format_exc())
            raise e
        return ret_val

    return inner
