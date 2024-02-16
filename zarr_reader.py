import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt


class ZarrReader(object):
    def __init__(self):
        reseviour = 'crimac-scratch'
        year = '2022'
        survey_number = 'S2022611'
        file_location = os.path.join('/Users/changkyu/GitHub', reseviour, year, survey_number, 'ACOUSTIC/GRIDDED')

        bottom_location = os.path.join(file_location, '%s_bottom.zarr' % survey_number)
        labels_location = os.path.join(file_location, '%s_labels.zarr' % survey_number)
        sv_location = os.path.join(file_location, '%s_sv.zarr' % survey_number)



        self.bottom = xr.open_zarr(bottom_location)
        self.labels = xr.open_zarr(labels_location)
        self.sv = xr.open_zarr(sv_location)

width = sv.ping_time.__len__()
depth = sv.range.__len__()
num_chunks = width//depth


chunk_idx = 0


def get_chunk():






chunk = sv.sv
