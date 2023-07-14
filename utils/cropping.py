import numpy as np


def get_bbox_crop(center, patch_size):
    patch_corners = np.array([[0, 0], [patch_size[0], patch_size[1]]])
    data_coord = patch_corners + np.array(center) - np.array(patch_size) // 2

    # (x0, y0), (x1, y1)
    return data_coord.astype(int)


def get_slice_coords(cruise, cruise_coords):
    num_pings = cruise.num_pings()
    num_ranges = cruise.num_ranges()

    start_ping = max(0, cruise_coords[0][0])
    end_ping = min(num_pings, cruise_coords[1][0])

    start_range = max(0, cruise_coords[0][1])
    end_range = min(num_ranges, cruise_coords[1][1])

    # (x0, y0), (x1, y1)
    return np.array([[start_ping, start_range], [end_ping, end_range]]).astype(int)


def get_crop_idxs(slice_coords, cruise_coords, patch_size):
    crop_idxs = [[slice_coords[0][0] - cruise_coords[0][0],
                  slice_coords[0][1] - cruise_coords[0][1]],
                 [patch_size[0] - (cruise_coords[1][0] - slice_coords[1][0]),
                  patch_size[1] - (cruise_coords[1][1] - slice_coords[1][1])]]
    # (x0, y0), (x1, y1)
    return np.array(crop_idxs).astype(int)


def crop_data(cruise, center_location, patch_size, frequencies=None, boundary_val=np.nan):
    if frequencies is None:
        frequencies = cruise.frequencies()

    cruise_coords = get_bbox_crop(center_location, patch_size)
    slice_coords = get_slice_coords(cruise, cruise_coords)

    data = cruise.get_sv_slice(slice_coords[0][0], slice_coords[1][0], slice_coords[0][1], slice_coords[1][1],
                               frequencies)

    out_data = np.ones(shape=(len(frequencies), patch_size[0], patch_size[1])) * boundary_val
    crop_idxs = get_crop_idxs(slice_coords, cruise_coords, patch_size)

    out_data[:, crop_idxs[0][0]:crop_idxs[1][0], crop_idxs[0][1]:crop_idxs[1][1]] = data.values
    return out_data


def crop_annotations(cruise, center_location, patch_size, categories=None, boundary_val=np.nan):
    if categories is None:
        categories = cruise.categories()

    cruise_coords = get_bbox_crop(center_location, patch_size)
    slice_coords = get_slice_coords(cruise, cruise_coords)

    annotation = cruise.get_annotation_slice(slice_coords[0][0], slice_coords[1][0], slice_coords[0][1], slice_coords[1][1],
                                    categories)

    out_labels = np.ones(shape=(len(categories), patch_size[0], patch_size[1])) * boundary_val
    crop_idxs = get_crop_idxs(slice_coords, cruise_coords, patch_size)

    out_labels[:, crop_idxs[0][0]:crop_idxs[1][0], crop_idxs[0][1]:crop_idxs[1][1]] = annotation.values
    return out_labels


def crop_bbox(cruise, center_location, patch_size, categories=None):
    if categories is None:
        categories = cruise.categories()

    cruise_coords = get_bbox_crop(center_location, patch_size)
    slice_coords = get_slice_coords(cruise, cruise_coords)

    df = cruise.get_school_boxes(slice_coords[0][0], slice_coords[1][0], slice_coords[0][1], slice_coords[1][1],
                                    categories)
    bboxes = df[['startpingindex', 'upperdeptindex', 'endpingindex', 'lowerdeptindex']].values.astype(np.int32)
    category_labels = df.category.values.astype(np.int32)

    # Crop bboxes to fit patch
    bboxes[bboxes[:, 0] < slice_coords[0][0], 0] = slice_coords[0][0]
    bboxes[bboxes[:, 1] < slice_coords[0][1], 1] = slice_coords[0][1]
    bboxes[bboxes[:, 2] >= slice_coords[1][0], 2] = slice_coords[1][0]
    bboxes[bboxes[:, 3] >= slice_coords[1][1], 3] = slice_coords[1][1]

    # Convert to patch coordinates
    bboxes[:, [0, 2]] -= cruise_coords[0][0]
    bboxes[:, [1, 3]] -= cruise_coords[0][1]

    return bboxes, category_labels
