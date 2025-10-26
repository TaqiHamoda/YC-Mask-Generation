import csv
import numpy as np
from typing import Dict, List


class Annotations:
    def __init__(self, csv_path):
        self.data: Dict[str, List[np.ndarray, np.ndarray]] = {}

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)

            for row in reader:
                if len(row) != 4:
                    continue

                img_name, x, y, cl = row
                img_name, cl = img_name.strip(), cl.strip()

                try:
                    x, y = round(float(x), 3), round(float(y), 3)
                except ValueError:
                    continue  # Skip if x or y is not a float

                if self.data.get(img_name) is None:
                    self.data[img_name] = [[cl], [(x, y)]]
                else:
                    self.data[img_name][0].append(cl)
                    self.data[img_name][1].append((x, y))

        for img_name in self.data.keys():
            self.data[img_name][0] = np.array(self.data[img_name][0])
            self.data[img_name][1] = np.array(self.data[img_name][1])
