import yaml

def load_config_yaml(nfile):
    """
    Loads configuration file
    """

    ext = ".yaml" if "yaml" not in nfile else ""

    f = open(nfile + ext, "r")
    return yaml.load(f, Loader=yaml.Loader)