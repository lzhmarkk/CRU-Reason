from pyhealth.models import Deepr, AdaCare, GRASP, Transformer, RETAIN, RNN
from baselines.DrugRec import MySafeDrug, MyGAMENet, MyMICRON, MyMoleRec, MyStageNet


def get_model(config, mode, feature_keys=None, full_dataset=None, datasets=None):
    if feature_keys is None:
        feature_keys = ["conditions", "procedures", "drugs"]

    kwargs = {"dataset": full_dataset,
              "feature_keys": feature_keys,
              "label_key": "drugs" if config.task == 'drugrec' else "label",
              "mode": mode,
              "embedding_dim": config.embedding_dim}

    method = config.method
    if method == "GRU":
        model = RNN(num_layers=1, **kwargs)
    elif method == "Transformer":
        model = Transformer(num_layers=1, **kwargs)
    elif method == "RETAIN":
        model = RETAIN(**kwargs)
    elif method == "GRAM":
        # todo
        raise ValueError()
    elif method == "Deepr":
        model = Deepr(**kwargs)
    elif method == "StageNet":
        model = MyStageNet(**kwargs)
    elif method == "AdaCare":
        assert config.task in ["mortality", "readmission"]
        model = AdaCare(use_embedding=[True, True, True], **kwargs)
    elif method == "GRASP":
        assert config.task in ["mortality", "readmission"]
        model = GRASP(use_embedding=[True, True, True], **kwargs)
    elif method == "SafeDrug":
        assert config.task == "drugrec"
        model = MySafeDrug(num_layers=1, **kwargs)
    elif method == "MICRON":
        assert config.task == "drugrec"
        model = MyMICRON(**kwargs)
    elif method == "GAMENet":
        assert config.task == "drugrec"
        model = MyGAMENet(num_layers=1, **kwargs)
    elif method == "MoleRec":
        assert config.task == "drugrec"
        model = MyMoleRec(**kwargs)
    elif method == 'GraphCare':
        from baselines.graphcare.graphcare import GraphCare
        model = GraphCare(max_visits=config.data_features['max_visits'],
                          config=config.__dict__,
                          **kwargs)
    elif method == 'EMERGE':
        from baselines.EMERGE.EMERGE import EMERGE
        kwargs['config'] = config.__dict__
        kwargs['train_set'] = datasets['train']
        kwargs['val_set'] = datasets['val']
        kwargs['test_set'] = datasets['test']
        model = EMERGE(**kwargs)
    elif method == 'TRANS':
        from baselines.TRANS.TRANS import TRANS
        kwargs['config'] = config.__dict__
        model = TRANS(**kwargs)
    elif method == 'CoT':
        from baselines.CoT.CoT import CoT
        kwargs['config'] = config.__dict__
        kwargs['test_set'] = datasets['test']
        model = CoT(**kwargs)
    elif method == 'KARE':
        from baselines.KARE.KARE import KARE
        kwargs['config'] = config.__dict__
        kwargs['test_set'] = datasets['test']
        model = KARE(**kwargs)
    elif method == 'MedRAG':
        from baselines.MedRAG.MedRAG import MedRAG
        kwargs['config'] = config.__dict__
        kwargs['test_set'] = datasets['test']
        model = MedRAG(**kwargs)
    elif method == 'MKGRank':
        from baselines.MKGRank.MKGRank import MKGRank
        kwargs['config'] = config.__dict__
        kwargs['test_set'] = datasets['test']
        model = MKGRank(**kwargs)
    elif method == 'MedReason':
        from baselines.MedReason.MedReason import MedReason
        kwargs['config'] = config.__dict__
        kwargs['test_set'] = datasets['test']
        model = MedReason(**kwargs)
    elif method == 'MedIKAL':
        from baselines.MedIKAL.MedIKAL import MedIKAL
        kwargs['config'] = config.__dict__
        kwargs['test_set'] = datasets['test']
        model = MedIKAL(**kwargs)
    elif method == 'RetCare':
        from baselines.RetCare.RetCare import RetCare
        kwargs['config'] = config.__dict__
        kwargs['test_set'] = datasets['test']
        model = RetCare(**kwargs)
    else:
        raise ValueError

    return model
