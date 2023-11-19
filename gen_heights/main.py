#!/usr/bin/python3

import numpy as np
from numpy.typing import NDArray
import scipy.interpolate as interp
import river_network
import json
import sys

# Scales values and stretches the heights map using a certain method of interpolation
#TODO implement cubic, by chopping and sitching
def scale(heights: NDArray, vertical_scale: float, horizontal_scale: int, method: str='linear') -> NDArray:
    heights /= heights.max()
    heights *= vertical_scale

    w,h = heights.shape
    grid = (np.linspace(0,w-1,w),np.linspace(0,h-1,h))
    sx, sy = np.meshgrid(
    np.linspace(0,w-1, horizontal_scale*w),
    np.linspace(0,h-1, horizontal_scale*h)
    )
    points = np.array([sx.ravel(),sy.ravel()]).T
    rgi = interp.RegularGridInterpolator(grid, heights, method=method)
    values = rgi(points)
    return values.reshape(horizontal_scale*h, horizontal_scale*w)


def main(argv):
    if len(argv) != 2:
        print('Usage: %s <output.csv>' % (argv[0],))
        sys.exit(-1)

    output_path = argv[1]

    config = None
    with open("config.json", "r") as f:
        config = json.load(f)

    print('Scaling and rendering CSV...')

    heights, _ = river_network.get_all(config["dim"], config["disc_radius"], config["max_delta"], config["river_downcutting_constant"],
                                        config["directional_inertia"], config["default_water_level"], config["evaporation_rate"])

    heights = scale(heights, config["vertical_scale"], config["horizontal_scale"])
    heights = heights.astype(np.uint16)
    heights += np.ones(heights.shape, dtype=np.uint16)*config["base_height"]

    np.savetxt(output_path, heights, delimiter=',', fmt='%lf')


if __name__ == '__main__':
  main(sys.argv)

