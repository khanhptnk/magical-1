import jsonargparse

def parse():

    parser = jsonargparse.ArgumentParser()

    parser.add_argument('-config', type=str)
    parser.add_argument('-seed', type=int, default=123)
    parser.add_argument('-exp_root_dir', type=str, default='experiments')
    parser.add_argument('-name', type=str)
    parser.add_argument('-device_id', type=int)
    parser.add_argument('-eval_mode', type=int)

    parser.add_argument('dataset.n_train', type=int, default=1)
    parser.add_argument('dataset.n_eval', type=int, default=1)

    flags = parser.parse_args()

    return flags.config, jsonargparse.namespace_to_dict(flags)
