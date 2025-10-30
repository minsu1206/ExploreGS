"""
Same code as 
"""
import yaml

class ConfigDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        self[name] = value
    
    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")

def convert_to_config_dict(d):
    if isinstance(d, dict):
        return ConfigDict({k: convert_to_config_dict(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [convert_to_config_dict(i) for i in d]
    else:
        return d
    
def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return convert_to_config_dict(config)


