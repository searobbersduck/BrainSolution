import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

import aneurysm_analysis.models.retina.retina_net as detect_net
import aneurysm_analysis.external_lib.medicaldetectiontoolkit.model.configs as configs
import aneurysm_analysis.external_lib.medicaldetectiontoolkit.utils.exp_utils as utils

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str,  default='train_test',
                        help='one out of: train / test / train_test / analysis / create_exp')
    parser.add_argument('-f','--folds', nargs='+', type=int, default=None,
                        help='None runs over all folds in CV. otherwise specify list of folds.')
    parser.add_argument('--exp_dir', type=str, default='/path/to/experiment/directory',
                        help='path to experiment dir. will be created if non existent.')
    parser.add_argument('--server_env', default=False, action='store_true',
                        help='change IO settings to deploy models on a cluster.')
    parser.add_argument('--slurm_job_id', type=str, default=None, help='job scheduler info')
    parser.add_argument('--use_stored_settings', default=False, action='store_true',
                        help='load configs from existing exp_dir instead of source dir. always done for testing, '
                             'but can be set to true to do the same for training. useful in job scheduler environment, '
                             'where source code might change before the job actually runs.')
    parser.add_argument('--resume_to_checkpoint', type=str, default=None,
                        help='if resuming to checkpoint, the desired fold still needs to be parsed via --folds.')
    parser.add_argument('--exp_source', type=str, default='experiments/toy_exp',
                        help='specifies, from which source experiment to load configs and data_loader.')
    parser.add_argument('-d', '--dev', default=False, action='store_true', help="development mode: shorten everything")

    args = parser.parse_args()
    return args

args = parse_args()

cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, args.use_stored_settings)

config = configs.configs()
print(dir(config))
print(args.cf)
# logger = utils.get_logger(cf.fold_dir)
# net = detect_net.net(config, )