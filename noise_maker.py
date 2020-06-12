from dataclasses import dataclass
from typing import (
    Any,
)

import numpy


@dataclass
class NoiseMaker:
    '''
    Presample and cache dirichlet noise
    '''
    num_samples_per_parameter: int = 10_000
    highest_index: int = None
    noise: Any = None

    def __post_init__(self):
        self.highest_index = self.num_samples_per_parameter - 1
        self.noise = {}

    def _sample_noise(self, alpha, size):
        return numpy.random.dirichlet([alpha] * size, self.num_samples_per_parameter).tolist()

    def make_noise(self, alpha, size):
        # Lookup cached noise
        # - If there is no previously cached noise then make some
        try:
            noise_data = self.noise[(alpha, size)]
        except KeyError:
            # Using an except avoids the "if not in" every call.  Normally I'd call "premature
            # optimization", but entire class exists to sample noise fast for an inner loop that
            # needs to be speedy.
            noise_data = [0, self._sample_noise(alpha, size)]
            self.noise[(alpha, size)] = noise_data

        # If your runway of noise has run out, cache some more.
        if noise_data[0] == self.highest_index:
            self.noise[(alpha, size)] = [0, self._sample_noise(alpha, size)]
        else:
            noise_data[0] = noise_data[0] + 1

        return noise_data[1][noise_data[0]]
