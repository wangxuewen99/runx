import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

default_summary_modules = ['Conv2d',  'Add', 'Linear', 'ReLU', 'BatchNorm2d', 'InstanceNorm2d', 'MaxPool2d', 'AvgPool2d', 'Upsample']


class SumX(object):
    def __init__(self, show_range=False, summary_modules=default_summary_modules, fusion=True):
        self.hooks = []
        self.layers = []
        self.summary_modules = summary_modules
        self.fusion = fusion
        self.id_to_name = {}
        self.show_range = show_range

    def _register_hook(self, module):
        handle = module.register_forward_hook(self._summary_hook)
        self.hooks.append(handle)

    def _summary_hook(self, module, input, output):
        module_type = module.__class__.__name__
        if module_type not in self.summary_modules:
            return
        epsilon = 1.0e-6

        if module_type == 'Conv2d' and not isinstance(module, nn.Conv2d):
            return

        if isinstance(output, tuple):
            output = output[0]

        if isinstance(module, nn.BatchNorm2d) and self.layers[-1]['type']=='Conv2d' and self.fusion==True:
            conv_weight = self.layers[-1]['weight']
            conv_bias = self.layers[-1]['bias']
            shift = module.bias
            scale = module.weight
            mean = module.running_mean
            var = module.running_var
            std = torch.sqrt(var + epsilon)

            if conv_bias is not None:
                conv_bias = shift + (conv_bias- mean) * scale / std
            else:
                conv_bias = shift - mean * scale / std
            conv_weight = conv_weight * (scale / std).view(-1, 1, 1, 1)
            self.layers[-1]['weight'] = conv_weight
            self.layers[-1]['bias'] = conv_bias
            self.layers[-1]['output'] =  output
            return

        name = self.id_to_name[str(id(module))]
        layer = OrderedDict()
        layer['name'] = name
        layer['type'] = module_type
        layer['input'] = input[0]
        layer['output'] = output

        param_num = 0
        flops = 0
        if hasattr(module, 'weight') and module.weight is not None:
            if isinstance(module, nn.Conv2d):
                _, _, output_height, output_width = output.size()
                output_channel, input_channel, kernel_height, kernel_width = module.weight.size()
                flops = output_channel * output_height * output_width * input_channel * kernel_height * kernel_width

            if isinstance(module, nn.Linear):
                input_num, output_num = module.weight.size()
                flops = input_num * output_num

            param_num += module.weight.numel()
            layer['trainable'] = module.weight.requires_grad
            layer['weight'] = module.weight
        else:
            layer['weight'] = None

        if hasattr(module, 'bias') and module.bias is not None:
            param_num += module.bias.numel()
            flops += module.bias.numel()
            layer['bias'] = module.bias
        else:
            layer['bias'] = None

        layer['param_num'] = param_num
        layer['flops'] = flops
        self.layers.append(layer)

    def summarize(self, model, *args, **kwargs):
        for name, module in model.named_modules():
            self.id_to_name[str(id(module))] = name

        # register hook
        model.eval()
        model.apply(self._register_hook)
        model(*args, **kwargs)

        self.print_info()
        if self.show_range:
            self.print_range()

        # clear instance information
        self.clear()

    def print_info(self):
        separator = '='*60
        print(separator)
        header = '{:<30}{:<12}{:<18}{:<18}{:<18}{:<12}{:<12}'.format('Layer', 'Type', 'Input', 'Output', 'Kernel', 'Params #', 'FLOPS #')
        print(header)

        print(separator)
        total_params = 0
        trainable_params = 0
        total_flops = 0
        for info in self.layers:
            input_shape = str(list(info['input'].size()))
            output_shape = str(list(info['output'].size()))
            kernel_shape = str(list(info['weight'].size())) if info['weight'] is not None else 'None'
            params = '{0:,}'.format(info['param_num'])
            flops = str('{:,}'.format(info['flops']))
            line = '{:<30}{:<12}{:<18}{:<18}{:<18}{:<12}{:<12}'.format(info['name'], info['type'], input_shape, output_shape, kernel_shape, params, flops)
            print(line)
            total_params += info['param_num']
            if 'trainable' in info and info['trainable'] == True:
                trainable_params += info['param_num']
            total_flops += info['flops']
        print(separator)
        print('Total params: {0:,}'.format(total_params))
        print('Trainable params: {0:,}'.format(trainable_params))
        print('Non-trainable params: {0:,}'.format(total_params - trainable_params))
        print('Total FLOPS: {0:,}'.format(total_flops))
        print(separator)

    def print_range(self):
        separator = '='*60
        print(separator)
        header = '{:<30}{:<12}{:<18}{:<18}{:<18}'.format('Layer', 'Type', 'Weight', 'Bias', 'Activation')
        print(header)

        print(separator)
        total_weights = []
        total_activations = []
        for info in self.layers:
            weight = '[{:.3f}, {:.3f}]'.format(info['weight'].min(), info['weight'].max())
            bias = '[{:.3f}, {:.3f}]'.format(info['bias'].min(), info['bias'].max())
            activation = '[{:.3f}, {:.3f}]'.format(info['output'].min(), info['output'].max())
            line = '{:<30}{:<12}{:<18}{:<18}{:<18}'.format(info['name'], info['type'], weight, bias, activation)
            print(line)

            total_weights.append(info['weight'].detach().cpu().numpy().reshape(-1))
            total_weights.append(info['bias'].detach().cpu().numpy().reshape(-1))
            total_activations.append(info['output'].detach().cpu().numpy().reshape(-1))

        print(separator)
        total_weights = np.concatenate(total_weights, axis=0)
        total_activations = np.concatenate(total_activations, axis=0)
        print('Total weights: {:.3f}, {:.3f}'.format(total_weights.min(), total_weights.max()))
        print('Total activations: {:.3f}, {:.3f}'.format(total_activations.min(), total_activations.max()))
        print(separator)

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.layers.clear()
        self.id_to_name.clear()
