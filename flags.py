import jsonargparse

def parse():

    parser = jsonargparse.ArgumentParser()

    parser.add_argument('-config', type=str)
    parser.add_argument('-seed', type=int, default=123)
    parser.add_argument('-exp_root_dir', type=str, default='experiments')
    parser.add_argument('-name', type=str)
    parser.add_argument('-device', type=int)
    parser.add_argument('-eval_mode', type=int)
    parser.add_argument('-data_dir', type=str)

    parser.add_argument('-env.resolution', type=str)

    parser.add_argument('-train.log_every', type=int)
    parser.add_argument('-train.batch_size', type=int)

    parser.add_argument('-policy.lr', type=float)
    parser.add_argument('-policy.load_from', type=str)

    parser.add_argument('-dataset.n_train', type=int, default=1000)
    parser.add_argument('-dataset.n_eval', type=int, default=100)

    flags = parser.parse_args()

    return flags.config, jsonargparse.namespace_to_dict(flags)
