from magical.experts.move_to_corner import MoveToCornerExpert
from magical.experts.move_to_region import MoveToRegionExpert

def load(config, *args):
    name = config.expert.name
    try:
        cls = globals()[name]
        return cls(*args)
    except KeyError:
        return Exception('No such expert: %s' % name)
