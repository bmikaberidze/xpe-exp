"""
Train and evaluate a model using the pipeline.

Usage:
    python -m scripts.models.load --config ./config/units/tune.xpe.lm.aya.ds.bebe.yml
    python -m scripts.models.load --config ./config/units/test.lm.aya.ds.bebe.yml
    
"""
if __name__ == '__main__':

    # Initiate micm nlp toolkit and get config from args
    from src.utils import micm_nlp_setup, pars_load_config
    # 
    micm_nlp_setup()
    config = pars_load_config()

    # Load Model
    from micm_nlp.pipeline import load_model
    # 
    model = load_model(config)

