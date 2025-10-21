import csv
import numpy as np
from typing import Dict, Tuple


class Annotations:
    def __init__(self, csv_path):
        self.data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)

            for row in reader:
                if len(row) != 4:
                    continue

                img_name, x, y, cl = row

                try:
                    x, y = round(float(x), 3), round(float(y), 3)
                except ValueError:
                    continue  # Skip if x or y is not a float

                if self.data.get(img_name) is None:
                    self.data[img_name] = (np.array([cl]), np.array([[x, y]]))
                else:
                    self.data[img_name][0] = np.vstack((self.data[img_name][0], [cl]))
                    self.data[img_name][1] = np.vstack((self.data[img_name][1], [[x, y]]))
