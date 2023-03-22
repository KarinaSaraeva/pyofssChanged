import numpy as np
from numpy import random
from pyofss.field import temporal_power

class Noise(object):
    def __init__(self, name="noise",
                 disp_factor=None):
        self.name = name
        self.disp_factor = disp_factor

    def __call__(self, domain, field):
        noise = random.normal(loc = 0, scale = np.sqrt(self.disp_factor*np.amax(temporal_power(field))), size=domain.t.shape)
        return field+noise