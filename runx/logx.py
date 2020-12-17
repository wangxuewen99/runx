import os
import yaml
from pathlib import Path
import shutil
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    from tensorboardX import SummaryWriter

def dict_to_md(d):
    s = ''
    for k, v in d.items():
        s += f'{k}: {v}  \n' # two space for new line

    return s

class LogX(SummaryWriter):
    def __init__(self, log_dir, name, hparams=None):
        self.log_dir = Path(log_dir, name)
        self.checkpoint_dir = self.log_dir / 'checkpoints'
        self.log_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.log_file = (self.log_dir / 'log.txt').open('a')

        super(LogX, self).__init__(self.log_dir)

        if hparams:
            if not isinstance(hparams, dict):
                hparams = vars(hparams)

            with (self.log_dir / 'config.yaml').open('w') as f:
                yaml.dump(hparams, f, sort_keys=False, width=200)

            if self.writer is not None:
                text = dict_to_md(hparams)
                self.add_text('hparams', text)

    def msg(self, msg):
        print(msg)
        self.log_file.write(msg + '\n')
        self.log_file.flush()

    def save_model(self, state_dict, idx):
        save_path = self.checkpoint_dir / f'checkpoint-{idx}.pth'
        torch.save(state_dict, save_path)
