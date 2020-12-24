from runx.config import BaseConfig

class Config(BaseConfig):
    lr = 0.01
    batch_size = 32
    input_size = (256, 256)
    weights = list(range(100))
    list_path = './data/list.txt'
