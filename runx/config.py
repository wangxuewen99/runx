import os.path as ospath
import copy
import json
import yaml
import sys
from importlib import import_module

def check_file_exist(filename):
    if not ospath.isfile(filename):
        raise FileNotFoundError(f'file {filename} dose not exist.')

class Config(object):
    def __init__(self, cfg_dict=None, filename=None):
        assert (cfg_dict is None and filename is not None) or (cfg_dict is not None and filename is None), \
            f'cfg_dict and filename only need one.'

        # new _cfg_dict as a container of configures
        object.__setattr__(self, '_cfg_dict', dict())
        if cfg_dict is not None:
            if not isinstance(cfg_dict, dict):
                raise TypeError(f'cfg_dict must be a dict, but got {type(cfg_dict)}')
            else:
                for key, value in cfg_dict.items():
                    self._cfg_dict[key] = value

        if filename is not None:
            self.load(filename)

    def to_dict(self):
        return self._cfg_dict

    def clear(self):
        self._cfg_dict.clear()

    def load(self, filename):
        check_file_exist(filename)
        suffix = ospath.splitext(filename)[-1]
        if suffix not in ['.py', '.json', '.yaml']:
            raise IOError('Only py/json/yaml type are supported now.')

        self.clear()    # clear Config status
        if suffix == '.json':
            with open(filename, 'r') as f:
                cfg_dict = json.load(f)
        elif suffix in ['.yaml', '.yml']:
            with open(filename, 'r') as f:
                cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        elif suffix == '.py':   # load configures from py file
            config_dir = ospath.dirname(filename)
            sys.path.insert(0, config_dir)
            module_name = ospath.splitext(ospath.basename(filename))[0]
            mod = import_module(module_name)
            sys.path.pop(0)
            cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__')
            }
            del sys.modules[module_name]
        else:
            pass

        for name, value in cfg_dict.items():
            self._cfg_dict[name] = value

    def dump(self, filename):
        suffix = ospath.splitext()[-1]
        if suffix not in ['.json', '.yaml', '.yml']:
            raise IOError('Only json/yaml type are supported now.')

        if suffix == '.json':
            with open(filename, 'w') as f:
                json.dump(self._cfg_dict, f, indent=4)

        elif suffix in ['.yaml', '.yml']:
            with open(filename, 'w') as f:
                yaml.dump(self._cfg_dict, f, sort_keys=False, width=256)
        else:
            pass

    def __len__(self):
        return len(self._cfg_dict)

    def __getitem__(self, name):
        return self._cfg_dict[name]

    def __setitem__(self, name, value):
        self._cfg_dict[name] = value

    def __iter__(self):
        return iter(self._cfg_dict.items())

    def __getattr__(self, name):
        return self._cfg_dict[name]

    def __setattr__(self, name, value):
        self._cfg_dict[name] = value

    def __getstate__(self):
        return self._cfg_dict

    def __setstate__(self, cfg_dict):
        for name, value in cfg_dict:
            self._cfg_dict[name] = value
