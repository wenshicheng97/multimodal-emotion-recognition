def get_search_hparams(config):
    search_hparams = []
    for k,v in config['parameters'].items():
        if 'values' in v:
            search_hparams.append(k)
    return search_hparams


def get_experiment_name(search_hparams, hparams):
    if search_hparams:
        return '_'.join([hparams.model_name] + [f'{k}={v}' for k,v in hparams.items() if k in search_hparams])
    return f'{hparams.model_name}'