"""test sinogram input module"""

import os
from os.path import join, dirname
from copy import deepcopy
import unittest
import tempfile
import mrcfile as mf
from numpy import testing as npt
import numpy as np
import h5py

from tests.testdata import datasets, experiments 


from clic.inout.sinogram_input import *
from clic.log import Config


class SinogramInputTest(unittest.TestCase):

    def setUp(self):
        
        self.orig_image = mf.read(join(dirname(datasets.__file__), "000_2cg9_particles_100.mrcs"))[0]

        self.sino= mf.read(join(dirname(datasets.__file__), 
                               "000_2cg9_sinograms_100.mrcs"))[0]

        self.lowpass = 10
        self.highpass = 50

        config_data = {
            "dataset": "particle_list.txt",
            "num": 1000,
            "batch_size": None,
            "downscale": 1,
            "filter_method": "butter",
            "lowpass": 5,
            "highpass": None,
            "pixel_size": 1,
            "snr": None,
            "model": "UMAP",
            "lines": 120,
            "comps": 10,
            "clusters": 3,
            "gpu":True,
            "save_model": False
            }

        self.config = Config(**config_data)

    def test_preprocess(self):

        sinogram,_ = preprocess(self.orig_image, self.config, self.orig_image.shape[0])

        npt.assert_array_equal(self.sino.shape, sinogram.shape)
        npt.assert_array_almost_equal(self.sino, sinogram.astype(np.float32))

    