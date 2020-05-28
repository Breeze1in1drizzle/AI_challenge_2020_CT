import os
abs_path = os.path.abspath(__file__)
# this variable is the root path in Eumpy
ROOT_PATH = abs_path.replace('\\', '/')[:-16]

DATA_PATH = ROOT_PATH + "datasets/"
