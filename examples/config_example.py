from runx.config import Config

cfg = Config(filename='./configs/config.py')

print(cfg.lr)
print(cfg.batch_size)
print(cfg.input_size)
print(cfg.weights)
print(cfg.list_path)
print(cfg['lr'])
print(cfg['batch_size'])
print(cfg['input_size'])
print(cfg['weights'])
print(cfg['list_path'])

