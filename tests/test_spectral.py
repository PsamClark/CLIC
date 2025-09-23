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
        
        self.orig_image = mf.read(join(dirname(datasets.__file__), "000_2cg9_particles_100.mrcs"))[0]

        self.butter_filt = mf.read(join(dirname(datasets.__file__), "000_2cg9_l10_h50_butter.mrc"))

        self.gauss_filt = mf.read(join(dirname(datasets.__file__), "000_2cg9_l10_h50_gauss.mrc"))

        self.butter_mask = mf.read(join(dirname(datasets.__file__), "l10_h50_butter_mask.mrc"))

        self.gauss_mask = mf.read(join(dirname(datasets.__file__), "l10_h50_gauss_mask.mrc"))

        self.outer = mf.read(join(dirname(datasets.__file__), "000_2cg9_image_outer.mrc"))

        self.fft2d = mf.read(join(dirname(datasets.__file__), "000_2cg9_2d_fft.mrc"))

        self.lowpass = 10
        self.highpass = 50


    def test_bandpass_image_fail(self):

        with self.assertRaises(ValueError):

            _,_ = bandpass_image(self.orig_image,low=None,high=None)


    def test_butter_filter(self):

        filter_image, filter_mask = bandpass_image(self.orig_image, low = self.lowpass,high = self.highpass)


        npt.assert_array_equal(self.butter_filt,filter_image)

        npt.assert_array_equal(self.butter_mask, filter_mask)

    def test_gauss_filter(self):

        filter_image, filter_mask = bandpass_image(self.orig_image, low = self.lowpass,high = self.highpass
                                                   method = "gauss")


        npt.assert_array_equal(self.gauss_filt,filter_image)

        npt.assert_array_equal(self.gauss_mask, filter_mask)



    def test_get_image_outer(self):
        
        outer_image = get_image_outer(self.orig_image)

        npt.assert_array_equal(self.outer,outer_image)

    

