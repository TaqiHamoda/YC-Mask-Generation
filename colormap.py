import csv
import numpy as np
import matplotlib.cm as cm
from typing import Dict, Tuple


class Colormap:
    hex_to_rgb = lambda h: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
    rgb_to_hex = lambda c: '#{:02x}{:02x}{:02x}'.format(c[0], c[1], c[2])
    get_random_rgb = lambda: tuple((255 * np.array(cm.gist_ncar(np.random.random())[:3])).astype(int))

    def __init__(self, csv_path):
        self.file_path = csv_path
        self.colormap: Dict[str, Tuple[int, int, int]] = {}

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)

            for row in reader:
                label, color = row
                self.colormap[label] = Colormap.hex_to_rgb(color)

    def updateColorMap(self):
        with open(self.file_path, 'w') as f:
            writer = csv.writer(f)
            for label, color in self.colormap.items():
                writer.writerow((label, Colormap.rgb_to_hex(color)))

    def getColor(self, label) -> Tuple[int, int, int]:
        if self.colormap.get(label) is None:
            rgba_color_float = Colormap.get_random_rgb()
            while rgba_color_float in self.colormap.values():
                rgba_color_float = Colormap.get_random_rgb()

            self.colormap[label] = rgba_color_float

            self.updateColorMap()

        return self.colormap[label]
