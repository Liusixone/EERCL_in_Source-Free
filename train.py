import os
from basicda.utils.train_api import train
from basicda.utils.arg_parser import basicda_arg_parser
from EERCL.models import *
from EERCL.loaders import *
from EERCL.trainers import *
from EERCL.models import *
from EERCL.loaders import *
from EERCL.trainers import *

if __name__ == '__main__':
    project_root = os.getcwd()
    package_name = 'EERCL'
    arg = basicda_arg_parser(project_root, package_name)
    train(arg)
    print('Done!')

