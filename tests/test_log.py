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

from clic.log import *


class LogTest(unittest.TestCase):


    def setUp(self):

        self._orig_dir = os.getcwd()
        self.temp_dir = tempfile.TemporaryDirectory()

        self.confile = join(dirname(experiments.__file__), "Configs/46lLtH.json")

        self.confile_missing_path = join(
            dirname(experiments.__file__),
            "Configs/config_missing_dpath.json")
        self.confile_wrong_type = join(
            dirname(experiments.__file__),
            "Configs/config_wrong_type.json")

        config_data = {
            "dataset": "particle_list.txt",
            "num": 1000,
            "batch_size": None,
            "downscale": 1,
            "filter_method": "butter",
            "lowpass": 19,
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

        self.default_config = Config()

        self.image_path = join(dirname(datasets.__file__), 
                               "000_2cg9_particles_100.mrcs")
        
        self.sino_path = join(dirname(datasets.__file__), 
                               "000_2cg9_sinograms_100.mrcs")

    def test_validate_config(self):

        data = load_config(
            self.confile
        )

        config_data = self.config

        self.assertEqual(
            len(data.model_dump().items()), len(self.default_config.model_dump().items())
        )

        self.assertEqual(data.model_dump(),config_data.model_dump())

    def tearDown(self):
        os.chdir(self._orig_dir)
        self.temp_dir.cleanup()

    def test_validate_config_fail(self):

        with self.assertRaises(ValueError):
            load_config(self.confile_missing_path)

        with self.assertRaises(ImportError):
            load_config(self.confile_wrong_type)
    def test_write_config_file(self):

    
        os.chdir(self.temp_dir.name)

        store_config(self.config,"test")

        files = glob(
            f"{self.temp_dir.name}/Configs/*"
        )

        data_from_output = load_config(files[0])

        self.assertEqual(len(files), 1)

        self.assertEqual(data_from_output.model_dump().items(), 
                         self.config.model_dump().items())

    def test_store_images(self):

        ims = mf.read(self.image_path)
        sinos = mf.read(self.sino_path)

        image_data = {'images':ims,
                      'sinograms':sinos,
                      'ids':np.array(["000"]*100)}

        store_images(ims,sinos,["000"]*100,self.temp_dir.name)

        imfile = h5py.File(f"{self.temp_dir.name}/batch0_images.hdf5","r")

        ids = imfile["ids"][:]
        npt.assert_array_equal(image_data['images'],imfile['images'])
        npt.assert_array_equal(image_data['sinograms'],imfile['sinograms'])
        npt.assert_array_equal(image_data['ids'],ids)



