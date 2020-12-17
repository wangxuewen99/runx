import torch
import numpy as np
import copy

class Meter(object):
    def reset(self):
        pass

    def add(self, value):
        pass

    def value(self):
        pass

def check_type(value):
    if isinstance(value, list):
        new = []
        for v in value:
            if isinstance(v, torch.Tensor):
                new.append(v.item())
            else:
                new.append(v)
    elif isinstance(value, dict):
        new = {}
        for k, v in value.items():
            if isinstance(v, torch.Tensor):
                new[k] = v.item()
            else:
                new[k] = v
    else:
        new = value
    return new

class AverageMeter(Meter):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = None

    # value can be a number, numpy array, list
    def add(self, value):
        value = check_type(value)

        if self.count == 0:
            self.sum = copy.deepcopy(value)
        elif isinstance(value, list):
            for k, v in enumerate(value):
                self.sum[k] += v
        elif isinstance(value, dict):
            for k, v in value.items():
                self.sum[k] += v
        else:
            self.sum += value
        self.count += 1

    def value(self):
        out = copy.deepcopy(self.sum)
        if isinstance(self.sum, list):
            for k, v in enumerate(self.sum):
                out[k] = v / float(self.count)
        elif isinstance(self.sum, dict):
            for k, v in self.sum.items():
                out[k] = v / float(self.count)
        else:
            out = self.sum / float(self.count)
        return out

class MovingAverageMeter(Meter):
    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.curr_value = None
        self.last_value = None
        self.count = 0

    # value can be a number, numpy array, list
    def add(self, value):
        value = check_type(value)

        if self.count == 0:
            self.curr_value = copy.deepcopy(value)
        elif isinstance(value, list):
            for k, v in enumerate(value):
                self.curr_value[k] = self.gamma*self.last_value[k] + (1.0 - self.gamma)*v
        elif isinstance(value, dict):
            for k, v in value.items():
                self.curr_value[k] = self.gamma*self.last_value[k] + (1.0 - self.gamma)*v
        else:
            self.curr_value = self.gamma * self.last_value + (1.0 - self.gamma)*value
        self.last_value = copy.deepcopy(self.curr_value)
        self.count +=1

    def value(self):
        return copy.deepcopy(self.curr_value)
