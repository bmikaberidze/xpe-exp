"""
Load Dataset
Example:
python -m scripts.datasets.preprocess --config ./config/proc.ds.xsc.tok.aya.yml
python -m scripts.datasets.preprocess --config ./config/proc.ds.xsc.tok.bloom.yml
python -m scripts.datasets.preprocess --config ./config/proc.ds.bebe.tok.aya.yml
"""

if __name__ == '__main__':
    
    import micm_nlp
    from micm_nlp.config import CONFIG
    from micm_nlp.pipeline import preprocess_dataset
    from src.utils import micm_nlp_setup

    micm_nlp_setup()

    # Parse Config Name Argument
    config_path = micm_nlp.utils.parse_script_args()
   
    # Load Configuration
    config = CONFIG.from_yaml(config_path)

    # Preprocess Dataset
    dataset = preprocess_dataset(config)
    
    
