import json
import numpy

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class BaseConfig():
    def to_dict(self):
        d = {}
        for key, value in vars(self.__class__).items():
            if not key.startswith('__') and key != 'to_dict':
                if callable(value):
                    d[key] = value.__name__
                elif isinstance(value, (numpy.ndarray,)):
                    d[key] = value.tolist()
                else:
                    d[key] = value
        return d

    def to_json(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent="    ")
            
    def load_json_config(self, path):
        with open(path, "r") as f:
            config = json.load(f)
        config = dotdict(config)
        return config
