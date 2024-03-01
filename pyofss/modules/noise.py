import numpy as np
from numpy import random
from pyofss.field import temporal_power, energy


class Noise(object):
    def __init__(self, name="noise", target_snr_db=None, Tr=None):
        self.name = name
        self.target_snr_db = target_snr_db
        self.Tr = Tr

    def __call__(self, domain, field):
        E = energy(field, domain.t)
        sig_avg = E / self.Tr if self.Tr is not None else E
        sig_avg_db = 10 * np.log10(sig_avg)
        noise_avg_db = sig_avg_db - self.target_snr_db

        noise_avg = 10 ** (noise_avg_db / 10)
        mean_noise = 0
        noise = np.random.normal(mean_noise, np.sqrt(noise_avg), np.shape(field))
        return field + noise
