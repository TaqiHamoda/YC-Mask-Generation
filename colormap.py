import csv
import numpy as np
import matplotlib.cm as cm
from typing import Dict, Tuple


class Colormap:
    hex_to_rgb = lambda h: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
    rgb_to_hex = lambda c: '#{:02x}{:02x}{:02x}'.format(c[0], c[1], c[2])

    def __init__(self, csv_path):
        self.file_path = csv_path
        self.colormap: Dict[str, Tuple[int, int, int]] = {}

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)

            for row in reader:
                label, color = row
                self.colormap[label] = self.hex_to_rgb(color)

    def updateColorMap(self):
        with open(self.file_path, 'w') as f:
            writer = csv.DictWriter(f)

            for label, color in self.colormap.items():
                writer.writerow({label: self.rgb_to_hex(color)})

    def getColor(self, label) -> Tuple[int, int, int]:
        if self.colormap.get(label) is None:
            rgba_color_float = cm.rainbow(np.random.random())
            while rgba_color_float not in self.colormap.values():
                rgba_color_float = cm.rainbow(np.random.random())

            self.colormap[label] = tuple((255 * np.array(rgba_color_float[:3])).astype(int))

            self.updateColorMap()

        return self.colormap[label]
