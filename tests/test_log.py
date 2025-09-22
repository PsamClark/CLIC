import os
from os.path import join, dirname
from copy import deepcopy
import unittest
import tempfile
import mrcfile as mf
import h5py

from tests.testdata.experiments import Configs
from tests.testdata import datasets

from clic.log import *


class LogTest(unittest.TestCase):


    def setUp(self):

        self._orig_dir = os.getcwd()
        self.temp_dir = tempfile.TemporaryDirectory()

        self.confile = join(dirname(Configs.__file__), "46lLtH.json")

        self.confile_missing_path = join(
            dirname(Configs.__file__),
            "config_missing_dpath.json")
        self.confile_wrong_type = join(
            dirname(Configs.__file__),
            "config_wrong_type.json")

        config_data = {
            "dataset": "particle_list.txt",
            "num": 3000,
            "batch_size": 1000,
            "downscale": 1,
            "filter_method": "butter",
            "lowpass": 7,
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
        
        self.config(**config_data)

        self.default_config = Config()

        self.image_path = join(dirname(datasets.__file__), 
                               "000_2cg9_particles_100.mrcs")
        
        self.sino_path = join(dirname(datasets.__file__), 
                               "000_2cg9_sinos_100.mrcs")

    def test_validate_config(self):

        data = load_config(
            self.confile
        )

        config_data = self.config.model_dump

        self.assertEqual(
            len(data.items()), len(self.default_mod.model_dump().items())
        )

        self.assertEqual(data,config_data)

    def tearDown(self):
        os.chdir(self._orig_dir)
        self.temp_dir.cleanup()

    def test_validate_config_fail(self):

        with self.assertRaises(ValueError):
            load_config(self.confile_missing_path)

        with self.assertRaises(TypeError):
            load_config(self.confile_wrong_type)

    def test_write_config_file(self):

    
        os.chdir(self.temp_dir.name)

        store_config(self.config)

        files = glob.glob(
            f"{self.temp_dir.name}/Configs/*"
        )

        data_from_output = load_config(config_file=files[0])

        self.assertEqual(len(files), 1)

        self.assertEqual(data_from_output, self.config.model_dump())

    def test_store_images(self):

        ims = mf.read(self.image_path)
        sinos = mf.read(self.sino_path)

        image_data = {'images':ims,
                      'sinograms':sinos,
                      'ids':["000"]*100}

        store_images(ims,sinos,["000"]*100,self.temp_dir.name)

        imfile = h5py.File(f"{self.temp_dir.name}/batch0_images.py","r")

        self.assertEqual(image_data,imfile)

