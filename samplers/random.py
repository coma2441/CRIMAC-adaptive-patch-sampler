import numpy as np

class Random:
    def __init__(self, cruise_list, num_samples=10000):
        self.cruise_list = cruise_list
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __call__(self, idx=None):
        random_cruise = np.random.choice(self.cruise_list)
        random_ping = np.random.randint(random_cruise.num_pings())
        random_range = np.random.randint(random_cruise.num_ranges())

        return {'cruise': random_cruise,
                'center_ping': random_ping,
                'center_range': random_range}





