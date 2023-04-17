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
    parser.add_argument('-use_wandb', type=int, default=0)
    parser.add_argument('-wandb_id', type=str)
    parser.add_argument('-debug_mode', type=int, default=0)

    parser.add_argument('-env.resolution', type=str)
    parser.add_argument('-env.view', type=str)

    parser.add_argument('-train.lr', type=float)
    parser.add_argument('-train.log_every', type=int)
    parser.add_argument('-train.batch_size', type=int)
    parser.add_argument('-train.eval_split', type=str)

    parser.add_argument('-policy.lr', type=float)
    parser.add_argument('-policy.load_from', type=str)

    parser.add_argument('-dataset.n_train', type=int, default=1000)
    parser.add_argument('-dataset.n_eval', type=int, default=100)
    parser.add_argument('-dataset.points_per_part', type=int)

    parser.add_argument('-dvae_dataset.trajs_per_model', type=int)
    parser.add_argument('-dvae.num_tokens', type=int)
    parser.add_argument('-dvae.codebook_dim', type=int)
    parser.add_argument('-dvae.hidden_dim', type=int)
    parser.add_argument('-dvae.max_kl_weight', type=float)
    parser.add_argument('-dvae.max_temp', type=float)

    flags = parser.parse_args()

    return flags.config, jsonargparse.namespace_to_dict(flags)
