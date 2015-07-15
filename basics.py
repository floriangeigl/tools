import os
import shutil


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
