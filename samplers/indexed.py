import pandas as pd

class Indexed:
    def __init__(self, cruise_list, index_file):
        self.cruise_list = cruise_list
        self.index_file = index_file
        self.index_df = pd.read_csv(self.index_file, index_col=0, header=0)

        self.cruise_name_index = {cruise.name: idx for idx, cruise in enumerate(self.cruise_list)}

    def __len__(self):
        return len(self.index_df)

    def __call__(self, idx):
        row = self.index_df.iloc[idx]

        cruise = self.cruise_list[self.cruise_name_index[row.cruise_name]]
        return {'cruise': cruise,
                'center_ping': int(row.ping_index),
                'center_range': int(row.range_index)}
