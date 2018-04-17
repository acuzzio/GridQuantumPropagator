import os
from contextlib import contextmanager

from quantumpropagator.errors import err


@contextmanager
def cd(newdir):
    '''
    This is used to enter in a folder with 'with cd(folder):' and send a command, then leave the folder.
    '''
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

def create_enumerated_folder(fn):
    '''
    this will create an incrementing folder.
    '''
    cont = 0
    while True:
        folder_path = '{}_{:04}'.format(fn,cont)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            break
        else:
            cont += 1

    return folder_path

def ensure_dir_Secure(folder_path):
    '''
    if the folder in folder_path does not exists it creates one
    This will abort if the folder is already there
    '''
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        err("The folder " + folder_path + " exists.")

def ensure_dir(folder_path):
    '''
    if the folder in folder_path does not exists it creates one
    '''
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

