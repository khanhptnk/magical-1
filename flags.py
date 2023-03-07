import jsonargparse

def parse():

    parser = jsonargparse.ArgumentParser()

    parser.add_argument('-config', type=str)
    parser.add_argument('-exp_root_dir', type=str, default='experiments')
    parser.add_argument('-name', type=str)
    parser.add_argument('-device_id', type=int)
    parser.add_argument('-view', type=str)
    parser.add_argument('-num_cpu', type=int)
    parser.add_argument('-eval_mode', type=int)


    parser.add_argument('-ent_coef', type=float, default=0.)

    flags = parser.parse_args()

    return flags.config, jsonargparse.namespace_to_dict(flags)
