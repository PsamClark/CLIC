import random
import string
from glob import glob
from pathlib import Path
import json
import pandas as pd
import h5py

from metrics import score_clustering

def random_string(length):

    """Generate random alphanumeric code of specified length
    """
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


def store_config(args,exp_id):
    
    """store configuration to json file
    """

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

    """store image information in hdf5 file
    """

    with h5py.File(f"{exp_id}/batch0_images.hdf5", "w") as imfile:

        imfile.create_dataset('batch0/images', data = all_ims)
        imfile.create_dataset('batch0/sinograms', data = all_sinos)
        imfile.create_dataset('batch0/ids', data = all_ids)

def collate_scores():

    """Collect scores and features
    """

    experiments = glob("Configs/*.json")
    out_list = []
    for exp_conf in experiments:

        exp_id = exp_conf.split("/")[-1][:-5]

        with open(exp_conf,"r") as conffile:

            config = json.load(conffile)

        score = score_clustering(exp_id)

        features, out_list = pop_features(
            config, score, out_list, exp_id
            )

    features.append("accuracy")
    features.insert(0,"exp_id")
    out_df = pd.DataFrame(data = out_list, columns=features)

    cwd = Path.cwd()

    out_df.to_csv(f"{cwd.name}_cs.csv", index=False)

def get_dict_entries(confdict):
    """Read dictionary keys and values
    """

    keys=[]
    values = []

    for k,v in confdict.items():

        if isinstance(v,dict):

            keys_new, values_new = get_dict_entries(v)

            keys_new_update = [k+'_'+i for i in keys_new]

            keys.extend(keys_new_update)
            values.extend(values_new)

        else:

            keys.append(k)
            values.append(v)

    return keys, values


def pop_features(config, score, out_list, exp_id):
    """populate features
    """
    features,values = get_dict_entries(config)
    print(score)
    values.append(score)
    values.insert(0, exp_id)

    out_list.append(values)

    return features, out_list

if __name__ == "__main__":

    collate_scores()
