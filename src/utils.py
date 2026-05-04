
from pathlib import Path
PROJECT_ROOT_PATH = Path(__file__).parent.parent

def micm_nlp_setup():
    import micm_nlp
    micm_nlp.init({
        'root_path': str(PROJECT_ROOT_PATH),
        'pretty_output': True,
    })

def pars_load_config():

    from micm_nlp.config import CONFIG
    from micm_nlp.utils import parse_script_args

    # Parse and Load Config
    config_path = parse_script_args()
    config = CONFIG.from_yaml(config_path)

    return config