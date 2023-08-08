import numpy as np

class Random:
    """ Sample randomly from a list of cruises """
    def __init__(self, cruise_list, num_samples=10000):
        """
        :param list[Cruise] cruise_list: list of cruises to sample from
        :param int num_samples: number of samples to draw, used to determine the length of a Dataset
        """
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





