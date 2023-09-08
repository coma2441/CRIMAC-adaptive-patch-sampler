import numpy as np


class Regular:
    def __init__(self, cruise_list, stride):
        """
        :param list[Cruise] cruise_list: list of cruises to sample from
        :param tuple patch_size: [width, height]
        :param int patch_overlap: number of pixels to overlap between neighboring patches
        """
        self.cruise_list = cruise_list
        self.stride = stride

        # Calculate center points
        self.regular_cruises = []
        self.cruise_idxs = []
        for i, cruise in enumerate(cruise_list):
            regular_cruise = RegularCruise(cruise, self.stride)
            self.regular_cruises.append(regular_cruise)
            self.cruise_idxs.extend([[i, idx] for idx in range(len(regular_cruise))])

    def __len__(self):
        return len(self.cruise_idxs)

    def __call__(self, idx):
        cruise_idx, grid_idx = self.cruise_idxs[idx]
        regular_cruise = self.regular_cruises[cruise_idx]
        return regular_cruise(grid_idx)


class RegularCruise:
    """ Sample from a cruise by iterating over a grid of patches """

    def __init__(self, cruise, stride=(256, 256)):
        """
        :param cruise: cruise object to sample from
        :param tuple stride: [horizontal, vertical] stride between center points
        """
        self.cruise = cruise
        self.stride = stride

        # Calculate center points of patches
        center_x = np.arange(self.stride[0] // 2, self.cruise.num_pings(), step=self.stride[0])
        center_y = np.arange(self.stride[1] // 2, self.cruise.num_ranges(), step=self.stride[1])
        self.centers = np.array(np.meshgrid(center_x, center_y)).T.reshape(-1, 2)

    def __len__(self):
        return len(self.centers)

    def __call__(self, idx):
        return {'cruise': self.cruise,
                'center_ping': self.centers[idx, 0],
                'center_range': self.centers[idx, 1]}
