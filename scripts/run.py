"""
Train and evaluate a model using the pipeline.

Usage:
    python -m scripts.run --config ./config/test.lm.bloom-7b1.ds.xsc.yml
    python -m scripts.run --config ./config/tune.xpe.lm.bloom.ds.xsc.yml
    #
    python -m scripts.run --config ./config/test.lm.aya.ds.xsc.yml
    python -m scripts.run --config ./config/tune.xpe.lm.aya.ds.xsc.yml
    # 
    python -m scripts.run --config ./config/test.lm.aya.ds.bebe.yml
    
"""
if __name__ == '__main__':

    # Initiate micm nlp toolkit and get config from args
    from src.utils import micm_nlp_setup, pars_load_config
    # 
    micm_nlp_setup()
    config = pars_load_config()

    # Run Pipline
    from micm_nlp.pipeline import run
    # 
    model, test_output = run(config)

