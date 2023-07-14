import numpy as np


class Gridded:
    def __init__(self, cruise_list, patch_size, patch_overlap=0):
        self.cruise_list = cruise_list
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap

        # Calculate center points
        self.gridded_cruises = []
        self.cruise_idxs = []
        for i, cruise in enumerate(cruise_list):
            gridded_cruise = GriddedCruise(cruise, patch_size, patch_overlap)
            self.gridded_cruises.append(gridded_cruise)
            self.cruise_idxs.extend([[i, idx] for idx in range(len(gridded_cruise))])

    def __len__(self):
        return len(self.cruise_idxs)

    def __call__(self, idx):
        cruise_idx, grid_idx = self.cruise_idxs[idx]
        gridded_cruise = self.gridded_cruises[cruise_idx]
        return gridded_cruise(grid_idx)


class GriddedCruise:
    def __init__(self, cruise, patch_size, patch_overlap=0):
        self.cruise = cruise
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap

        # Calculate center points of patches
        (patch_width, patch_height) = patch_size
        left = np.arange(-patch_overlap, self.cruise.num_pings() - patch_width, step=patch_width - 2 * patch_overlap)
        upper = np.arange(-patch_overlap, self.cruise.num_ranges() - patch_height, step=patch_height - 2 * patch_overlap)
        center_pings = left + patch_width // 2
        center_ranges = upper + patch_height // 2

        self.centers = np.array(np.meshgrid(center_pings, center_ranges)).T.reshape(-1, 2)

    def __len__(self):
        return len(self.centers)

    def __call__(self, idx):
        return {'cruise': self.cruise,
                'center_ping': self.centers[idx, 0],
                'center_range': self.centers[idx, 1]}

