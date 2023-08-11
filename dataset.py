import numpy as np
from utils.cropping import crop_data, crop_annotations, crop_bbox


class Base:
    def __init__(self, samplers, patch_size, frequencies, categories=None,
                 data_augmentation=None, data_transform=None, label_transform=None, sampling_probabilities=None,
                 num_samples=None):
        """
        Base dataset for marine acoustic data
        :param list samplers: list of sampler objects
        :param tuple patch_size: size of output patches, (width, height)
        :param list frequencies: list of acoustic frequencies to sample from
        :param list categories: list of categories to sample from
        :param data_augmentation:
        :param data_transform:
        :param label_transform:
        :param (list, None) sampling_probabilities: list of probabilities for each sampler, if None the probability is the same for all samplers
        :param (int, None) num_samples: number of samples to draw from the dataset, if None use total length
         of all samplers
        """

        self.samplers = samplers
        self.patch_size = patch_size
        self.frequencies = frequencies
        self.categories = categories
        self.data_augmentation = data_augmentation
        self.data_transform = data_transform
        self.label_transform = label_transform

        # Set num_samples, which defines the length of the dataset
        self.num_samples = num_samples if num_samples is not None else sum([len(sampler) for sampler in self.samplers])

        # Normalize sampling probabilities
        if sampling_probabilities is None:
            self.sampler_probs = np.ones(len(samplers))
        self.sampler_probs = np.array(self.sampler_probs)
        self.sampler_probs = np.cumsum(self.sampler_probs).astype(float)
        self.sampler_probs /= np.max(self.sampler_probs)

    def __len__(self):
        return self.num_samples

    def _get_location(self, idx):
        if len(self.samplers) > 1:
            i = np.random.rand()
            sampler = self.samplers[np.where(i < self.sampler_probs)[0][0]]
            out = sampler()
        else:
            sampler = self.samplers[0]
            out = sampler(idx)

        return out


class DatasetSegmentation(Base):
    def __init__(self, samplers, patch_size, frequencies, categories=None,
                 data_augmentation=None, data_transform=None, label_transform=None, sampling_probabilities=None,
                 num_samples=None):
        """
        Dataset for segmentation tasks
        :param list samplers: list of sampler objects
        :param tuple patch_size: size of output patches, (width, height)
        :param list frequencies: list of acoustic frequencies to sample from
        :param list categories: list of categories to sample from
        :param data_augmentation:
        :param data_transform:
        :param label_transform:
        :param (list, None) sampling_probabilities: list of probabilities for each sampler, if None the probability is the same for all samplers
        :param (int, None) num_samples: number of samples to draw from the dataset, if None use total length of all samplers
        """
        super().__init__(samplers, patch_size, frequencies, categories,
                         data_augmentation, data_transform, label_transform, sampling_probabilities, num_samples)

    def compute_segmentation_mask(self, labels):
        mask = np.zeros(self.patch_size)

        if self.categories is None:
            categories = np.arange(labels.shape[0]) + 1
        else:
            categories = self.categories

        for i, cat in enumerate(categories):
            mask[labels[i] == 1] = cat

        return mask.astype(int)

    def __getitem__(self, idx):
        sampler_out = self._get_location(idx)

        cruise = sampler_out['cruise']
        center_ping = sampler_out['center_ping']
        center_range = sampler_out['center_range']

        # do data crop
        center_location = np.array([center_ping, center_range])
        data = crop_data(cruise, center_location, self.patch_size, self.frequencies)
        annotation = crop_annotations(cruise, center_location, self.patch_size, self.categories)

        # Compute 2D segmentation mask from the labels
        mask = self.compute_segmentation_mask(annotation)

        if self.data_augmentation is not None:
            data, mask = self.data_augmentation(data, mask)
        if self.label_transform is not None:
            data, mask = self.label_transform(data, mask)
        if self.data_transform is not None:
            data, mask = self.data_transform(data, mask)

        return {'data': data, 'mask': mask, 'labels': annotation}


class DatasetBoundingBox(Base):
    def __init__(self, samplers, patch_size, frequencies,
                 data_augmentation=None, data_transform=None, label_transform=None, sampling_probabilities=None):
        """
        Dataset for bounding box tasks
        :param list samplers: list of sampler objects
        :param tuple patch_size: size of output data patches, (width, height)
        :param list frequencies: list of acoustic frequencies to sample from
        :param data_augmentation:
        :param data_transform:
        :param label_transform:
        :param (list, None) sampling_probabilities: list of probabilities for each sampler, if None the probability is the same for all samplers
        """
        super().__init__(samplers, patch_size, frequencies,
                         data_augmentation, data_transform, label_transform, sampling_probabilities)

    def __getitem__(self, idx):
        sampler_out = self._get_location(idx)

        cruise = sampler_out['cruise']
        center_ping = sampler_out['center_ping']
        center_range = sampler_out['center_range']

        # do data crop
        center_location = np.array([center_ping, center_range])
        data = crop_data(cruise, center_location, self.patch_size, self.frequencies)
        boxes, labels = crop_bbox(cruise, center_location, self.patch_size, self.categories)

        # TODO add transforms
        if self.data_augmentation is not None:
            data, boxes = self.data_augmentation(data, boxes)
        if self.label_transform is not None:
            data, boxes = self.label_transform(data, boxes)
        if self.data_transform is not None:
            data, boxes = self.data_transform(data, boxes)

        return {'data': data, 'boxes': boxes, 'labels': labels}
