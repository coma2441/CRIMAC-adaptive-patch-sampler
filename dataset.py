import numpy as np
from utils.cropping import crop_data, crop_labels

class Base:
    def __init__(self, samplers, patch_size, frequencies, categories=None,
                 data_augmentation=None, data_transform=None, label_transform=None, sampling_probabilities=None,
                 num_samples=None):
        self.samplers = samplers
        self.patch_size = patch_size
        self.frequencies = frequencies
        self.categories = categories
        self.data_augmentation = data_augmentation
        self.data_transform = data_transform
        self.label_transform = label_transform
        self.num_samples = num_samples if num_samples is not None else sum([len(sampler) for sampler in self.samplers])

        # Normalize sampling probabilities
        if sampling_probabilities is None:
            self.sampler_probs = np.ones(len(samplers))

        self.sampler_probs = np.array(self.sampler_probs)
        self.sampler_probs = np.cumsum(self.sampler_probs).astype(float)
        self.sampler_probs /= np.max(self.sampler_probs)

    def __len__(self):
        return self.num_samples

    def _get_sampler(self, idx):
        i = np.random.rand()
        sampler = self.samplers[np.where(i < self.sampler_probs)[0][0]]
        return sampler


class DatasetSegmentation(Base):
    def __init__(self, samplers, patch_size, frequencies,
                 data_augmentation=None, data_transform=None, label_transform=None, sampling_probabilities=None):
        super().__init__(samplers, patch_size, frequencies,
                 data_augmentation, data_transform, label_transform, sampling_probabilities)

    def __getitem__(self, idx):
        sampler = self._get_sampler(idx)
        sampler_out = sampler()

        cruise = sampler_out['cruise']
        center_ping = sampler_out['center_ping']
        center_range = sampler_out['center_range']

        # do data crop
        center_location = np.array([center_ping, center_range])
        data = crop_data(cruise, center_location, self.patch_size, self.frequencies)
        labels = crop_labels(cruise, center_location, self.patch_size, self.categories)

        # TODO create label mask


        if self.data_augmentation is not None:
            data, labels = self.data_augmentation(data, labels)
        if self.label_transform is not None:
            data, labels = self.label_transform(data, labels)
        if self.data_transform is not None:
            data, labels = self.data_transform(data, labels)

        return {'data': data, 'labels': labels}


class DatasetBoundingBox(Base):
    def __init__(self, samplers, patch_size, frequencies,
                 data_augmentation=None, data_transform=None, label_transform=None, sampling_probabilities=None):
        super().__init__(samplers, patch_size, frequencies,
                         data_augmentation, data_transform, label_transform, sampling_probabilities)

    def __getitem__(self, idx):
        sampler = self._get_sampler(idx)
        sampler_out = sampler()

        cruise = sampler_out['cruise']
        center_ping = sampler_out['center_ping']
        center_range = sampler_out['center_range']

        # do data crop
        center_location = np.array([center_ping, center_range])
        data = crop_data(cruise, center_location, self.patch_size, self.frequencies)
        labels = crop_labels(cruise, center_location, self.patch_size, self.categories)

        # TODO create bbox
        bbox = None

        if self.data_augmentation is not None:
            data, bbox = self.data_augmentation(data, bbox)
        if self.label_transform is not None:
            data, bbox = self.label_transform(data, bbox)
        if self.data_transform is not None:
            data, bbox = self.data_transform(data, bbox)

        return {'data': data, 'bbox': None}