import os
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