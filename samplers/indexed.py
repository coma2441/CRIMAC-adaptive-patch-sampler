import pandas as pd

class Indexed:
    """Sample based on a pre-computed index file with ping and range indices"""
    def __init__(self, cruise_list, index_file):
        """
        :param list cruise_list: list of cruises to sample from
        :param str index_file: path to index file
        """
        self.cruise_list = cruise_list
        self.index_file = index_file
        self.index_df = self._read_index_file()

        self.cruise_name_index = {cruise.name: idx for idx, cruise in enumerate(self.cruise_list)}

    def _read_index_file(self):
        try:
            return pd.read_csv(self.index_file, index_col=0, header=0)
        except FileNotFoundError:
            print("Index file not found at '{}'".format(self.index_file))
            raise

    def __len__(self):
        return len(self.index_df)

    def __call__(self, idx):
        row = self.index_df.iloc[idx]

        cruise = self.cruise_list[self.cruise_name_index[row.cruise_name]]
        return {'cruise': cruise,
                'center_ping': int(row.ping_index),
                'center_range': int(row.range_index)}
