from .arguments import *
import yaml

# Arguments parser for custom-defined argument dataclasses.

class customparser:

    def __init__(self, file_path):
        self.arg = None
        with open(file_path) as f:
            self.arg = yaml.safe_load(f)
            f.close()

    def parse_custom_args(self):
        return (circuitarguments(num_qubit      = self.arg['num_qubit'], \
                                freq            = self.arg['freq'], \
                                t2              = self.arg['t2'], \
                                gamma           = self.arg['gamma'], \
                                t_obs           = self.arg['t_obs'], \
                                num_points      = self.arg['num_points']), \
                optarguments(   opt             = self.arg['opt'], \
                                steps_per_point = self.arg['steps_per_point'], \
                                patience        = self.arg['patience'], \
                                threshold       = self.arg['threshold'], \
                                freq            = self.arg['freq'], \
                                use_l1          = self.arg['use_l1']), \
                otherarguments( save_to         = self.arg['save_to']))