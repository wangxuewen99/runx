import json
from pathlib import Path
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

        # initialize tensorboard
        super(LogX, self).__init__(self.log_dir)

        # backup hyper-parameters and show it in tensorboard txt tab
        if hparams:
            if not isinstance(hparams, dict):
                hparams = vars(hparams)

            with (self.log_dir / 'config.json').open('w') as f:
                json.dump(hparams, f, indent="    ")

            text = dict_to_md(hparams)
            self.add_text('hparams', text)

    # show message to screen and write to log file
    def msg(self, msg):
        print(msg)
        self.log_file.write(msg + '\n')
        self.log_file.flush()

    # save checkpoint to checkpoint directory name with 'checkpoint-idx.pth'
    def save_model(self, state_dict, idx):
        save_path = self.checkpoint_dir / f'checkpoint-{idx}.pth'
        torch.save(state_dict, save_path)
