from runx.config import Config

if __name__ == '__main__':
    config = Config(filename='./configs/config.py')
    print(config.to_dict())
    print(config['a'], config['b'])
    config['a'] = 5
    config['b'] = 7
    config['d'] = 10
    for a, b in config:
        print(a, b)

    print(config.a, config.b, config.d)
    config.c = [1, 2, 3]
    print(config.a, config.b, config.d, config.c)
    print(config.to_dict())
