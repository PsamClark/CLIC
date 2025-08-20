import random, string
from pathlib import Path
import json
import h5py


def random_string(length):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


def store_config(args,exp_id):

    Path("Configs").mkdir(exist_ok=True)
    config = {'model':{},'data':{},'preprocess':{}}

    config['data']['dir'] = args.data_set
    config['data']['dsize'] = args.num
    config['data']['bsize'] = args.batch_size

    config['preprocess']['downscale'] = args.down_scale
    config['preprocess']['filter_method'] = args.filter_method
    config['preprocess']['lowpass'] = args.lowpass
    config['preprocess']['highpass'] = args.highpass
    config['preprocess']['snr'] = args.snr

    config['model']['model'] = args.model
    config['model']['nlines'] = args.nlines
    config['model']['dims'] = args.num_comps
    config['model']['clusters'] = args.num_clusters

    with open(f"Configs/{exp_id}.json","w") as confile:

        json.dump(config,confile)

def store_images(all_ims, all_sinos, all_ids, exp_id):

    with h5py.File(f"{exp_id}/batch0_images.hdf5", "w") as imfile:

        imfile.create_dataset('batch0/images', data = all_ims)
        imfile.create_dataset('batch0/sinograms', data = all_sinos)
        imfile.create_dataset('batch0/ids', data = all_ids)

