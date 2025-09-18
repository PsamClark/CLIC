import os
from os.path import join, dirname
import unittest
import tempfile

import Configs
from tests import testdata
from clic.log import *


class LogTest(unittest.TestCase):


    def setUp(self):

        self._orig_dir = os.getcwd()
        self.temp_dir = tempfile.TemporaryDirectory()

        self.confile = join(dirname(Configs.__file__), "test_config.json")

        self.config_data = {
            'data':{
                'dir': join(dirname(testdata.__file__),"*.mrc"),
                'dsize':1000,
                'bsize':100
        },
        'preprocess':{
            'downscale':1,
            'filter_method':'butter',
            'lowpass':5, 
            'highpass':50,
            'snr': -1
        },
        'model':{
            'model':"UMAP",
            'nlines': 120,
            'dims':10,
            'clusters':3

        }
        }

        
        
    