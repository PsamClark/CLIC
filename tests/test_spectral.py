"""test spectral module"""

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


from clic.utils.spectral import *


class SpectralTest(unittest.TestCase):

    def setUp(self):
        
        self.orig_image = mf.read(join(dirname(datasets.__file__), "000_2cg9_particles_100.mrcs"))

        self.butter_filt = mf.read(join(dirname(datasets.__file__), "000_2cg9_l10_h50_butter.mrc"))

        self.gauss_filt = mf.read(join(dirname(datasets.__file__), "000_2cg9_l10_h50_gauss.mrc"))

        self.butter_mask = mf.read(join(dirname(datasets.__file__), "l10_h50_butter_mask.mrc"))

        self.gauss_mask = mf.read(join(dirname(datasets.__file__), "l10_h50_gauss_mask.mrc"))

        self.fft2d = mf.read(join(dirname(datasets.__file__), "000_2cg9_2d_fft.mrc"))

        self.lowpass = 10
        self.highpass = 50


    def test_butter_filter(self):

        filt_image, filt_mask = filter_image(self.orig_image, low = self.lowpass,high = self.highpass)

        npt.assert_array_equal(self.butter_mask, filt_mask)
        npt.assert_array_equal(self.butter_filt,filt_image)

  

    def test_gauss_filter(self):

        filt_image, filt_mask = filter_image(self.orig_image, low = self.lowpass, high = self.highpass,
                                                   method = "gauss")

        npt.assert_array_equal(self.gauss_mask, filt_mask)

        npt.assert_array_equal(self.gauss_filt,filt_image)






    

