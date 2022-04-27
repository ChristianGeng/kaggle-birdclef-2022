import os
from utils.utils import get_project_root
import json

def SCORED_BIRDS():
    fname = os.path.join(get_project_root(), "data", "interim", "scored_birds.json")
    with open(fname) as fp:
        SCORED_BIRDS = json.load(fp)
    return SCORED_BIRDS
