from yaml import safe_load


def load(file):

    """ Load .yaml file """

    with open(f"./Settings/{file}", "r") as stream:
        d = safe_load(stream)

    return d