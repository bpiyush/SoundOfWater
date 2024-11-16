"""Path utils."""
from os.path import dirname, abspath


curr_filepath = abspath(__file__)
repo_path = dirname(dirname(dirname(curr_filepath)))


def get_data_root_from_hostname():
    import socket

    data_root_lib = {
        "diva": "/ssd/pbagad/datasets/",
        "node": "/var/scratch/pbagad/datasets/",
        "fs4": "/var/scratch/pbagad/datasets/",
        "vggdev21": "/scratch/shared/beegfs/piyush/datasets/",
        "node407": "/var/scratch/pbagad/datasets/",
        "gnodee5": "/scratch/shared/beegfs/piyush/datasets/",
        "gnodeg2": "/scratch/shared/beegfs/piyush/datasets/",
        "gnodec2": "/scratch/shared/beegfs/piyush/datasets/",
        "Piyushs-MacBook-Pro": "/Users/piyush/projects/",
        "gnodec1": "/scratch/shared/beegfs/piyush/datasets/",
        "gnodec5": "/scratch/shared/beegfs/piyush/datasets/",
        "gnodec4": "/scratch/shared/beegfs/piyush/datasets/",
        "gnoded2": "/scratch/shared/beegfs/piyush/datasets/",
    }
    hostname = socket.gethostname()
    hostname = hostname.split(".")[0]
    
    assert hostname in data_root_lib.keys(), \
        "Hostname {} not in data_root_lib".format(hostname)

    data_root = data_root_lib[hostname]
    return data_root
    
