#!/usr/bin/python3

# MIT License
#
# Copyright (c) 2018 Daniel Andrino
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import collections
import heapq
import numpy as np
import matplotlib
import matplotlib.tri
import scipy as sp
import skimage.measure
import sys


# Returns the index of the smallest value of `a`
def min_index(a): return a.index(min(a))


# Returns an array with a bump centered in the middle of `shape`. `sigma`
# determines how wide the bump is.
def bump(shape, sigma):
  [y, x] = np.meshgrid(*map(np.arange, shape))
  r = np.hypot(x - shape[0] / 2, y - shape[1] / 2)
  c = min(shape) / 2
  return np.tanh(np.maximum(c - r, 0.0) / sigma)


# Returns a list of heights for each point in `points`.
def compute_height(points, neighbors, deltas, get_delta_fn=None):
  if get_delta_fn is None:
    get_delta_fn = lambda src, dst: deltas[dst]

  dim = len(points)
  result = [None] * dim
  seed_idx = min_index([sum(p) for p in points])
  q = [(0.0, seed_idx)]

  while len(q) > 0:
    (height, idx) = heapq.heappop(q)
    if result[idx] is not None: continue
    result[idx] = height
    for n in neighbors[idx]:
      if result[n] is not None: continue
      heapq.heappush(q, (get_delta_fn(idx, n) + height, n))
  return normalize(np.array(result))


# Same as above, but computes height taking into account river downcutting.
# `max_delta` determines the maximum difference in neighboring points (to
# give the effect of talus slippage). `river_downcutting_constant` affects how
# deeply rivers cut into terrain (higher means more downcutting).
def compute_final_height(points, neighbors, deltas, volume, upstream,
                         max_delta, river_downcutting_constant):
  dim = len(points)
  result = [None] * dim
  seed_idx = min_index([sum(p) for p in points])
  q = [(0.0, seed_idx)]

  def get_delta(src, dst):
    v = volume[dst] if (dst in upstream[src]) else 0.0
    downcut = 1.0 / (1.0 + v ** river_downcutting_constant)
    return min(max_delta, deltas[dst] * downcut)

  return compute_height(points, neighbors, deltas, get_delta_fn=get_delta)


# Computes the river network that traverses the terrain.
#   Arguments:
#   * points: The (x,y) coordinates of each point
#   * neghbors: Set of each neighbor index for each point.
#   * heights: The height of each point.
#   * land: Indicates whether each point is on land or water.
#   * directional_interta: indicates how straight the rivers are
#       (0 = no directional inertia, 1 = total directional inertia).
#   * default_water_level: How much water is assigned by default to each point
#   * evaporation_rate: How much water is evaporated as it traverses from along
#       each river edge.
#
#  Returns a 3-tuple of:
#  * List of indices of all points upstream from each point
#  * List containing the index of the point downstream of each point.
#  * The water volume of each point.
def compute_river_network(points, neighbors, heights, land,
                          directional_inertia, default_water_level,
                          evaporation_rate):
  num_points = len(points)

  # The normalized vector between points i and j
  def unit_delta(i, j):
    delta = points[j] - points[i]
    return delta / np.linalg.norm(delta)

  # Initialize river priority queue with all edges between non-land points to
  # land points. Each entry is a tuple of (priority, (i, j, river direction))
  q = []
  roots = set()
  for i in range(num_points):
    if land[i]: continue
    is_root = True
    for j in neighbors[i]:
      if not land[j]: continue
      is_root = True
      heapq.heappush(q, (-1.0, (i, j, unit_delta(i, j))))
    if is_root: roots.add(i)

  # Compute the map of each node to its downstream node.
  downstream = [None] * num_points

  while len(q) > 0:
    (_, (i, j, direction)) = heapq.heappop(q)

    # Assign i as being downstream of j, assuming such a point doesn't
    # already exist.
    if downstream[j] is not None: continue
    downstream[j] = i

    # Go through each neighbor of upstream point j.
    for k in neighbors[j]:
      # Ignore neighbors that are lower than the current point, or who already
      # have an assigned downstream point.
      if (heights[k] < heights[j] or downstream[k] is not None
          or not land[k]):
        continue

      # Edges that are aligned with the current direction vector are
      # prioritized.
      neighbor_direction = unit_delta(j, k)
      priority = -np.dot(direction, neighbor_direction)

      # Add new edge to queue.
      weighted_direction = lerp(neighbor_direction, direction,
                                     directional_inertia)
      heapq.heappush(q, (priority, (j, k, weighted_direction)))


  # Compute the mapping of each node to its upstream nodes.
  upstream = [set() for _ in range(num_points)]
  for i, j in enumerate(downstream):
    if j is not None: upstream[j].add(i)

  # Compute the water volume for each node.
  volume = [None] * num_points
  def compute_volume(i):
    if volume[i] is not None: return
    v = default_water_level
    for j in upstream[i]:
      compute_volume(j)
      v += volume[j]
    volume[i] = v * (1 - evaporation_rate)

  for i in range(0, num_points): compute_volume(i)

  return (upstream, downstream, volume)


# Renders `values` for each triangle in `tri` on an array the size of `shape`.
def render_triangulation(shape, tri, values):
  points = make_grid_points(shape)
  triangulation = matplotlib.tri.Triangulation(
      tri.points[:,0], tri.points[:,1], tri.simplices)
  interp = matplotlib.tri.LinearTriInterpolator(triangulation, values)
  return interp(points[:,0], points[:,1]).reshape(shape).filled(0.0)


# Removes any bodies of water completely enclosed by land.
def remove_lakes(mask):
  labels = skimage.measure.label(mask)
  new_mask = np.zeros_like(mask, dtype=bool)
  labels = skimage.measure.label(~mask, connectivity=1)
  new_mask[labels != labels[0, 0]] = True
  return new_mask


def get_all(dim: int, disc_radius: float, max_delta: float, river_downcutting_constant: float, directional_inertia: float, default_water_level: float, evaporation_rate: float):
  shape = (dim,) * 2
  
  print ('Generating heightmap...')

  print('  ...initial terrain shape')
  land_mask = remove_lakes(
      (fbm(shape, -2, lower=2.0) + bump(shape, 0.2 * dim) - 1.1) > 0)
  coastal_dropoff = np.tanh(dist_to_mask(land_mask) / 80.0) * land_mask
  mountain_shapes = fbm(shape, -2, lower=2.0, upper=np.inf)
  initial_height = (
      (gaussian_blur(np.maximum(mountain_shapes - 0.40, 0.0), sigma=5.0)
        + 0.1) * coastal_dropoff)
  deltas = normalize(np.abs(gaussian_gradient(initial_height)))

  print('  ...sampling points')
  points = poisson_disc_sampling(shape, disc_radius)
  coords = np.floor(points).astype(int)


  print('  ...delaunay triangulation')
  tri = sp.spatial.Delaunay(points)
  (indices, indptr) = tri.vertex_neighbor_vertices
  neighbors = [indptr[indices[k]:indices[k + 1]] for k in range(len(points))]
  points_land = land_mask[coords[:, 0], coords[:, 1]]
  points_deltas = deltas[coords[:, 0], coords[:, 1]]

  print('  ...initial height map')
  points_height = compute_height(points, neighbors, points_deltas)

  print('  ...river network')
  (upstream, downstream, volume) = compute_river_network(
      points, neighbors, points_height, points_land,
      directional_inertia, default_water_level, evaporation_rate)

  print('  ...final terrain height')
  new_height = compute_final_height(
      points, neighbors, points_deltas, volume, upstream,
      max_delta, river_downcutting_constant)
  terrain_height = render_triangulation(shape, tri, new_height)

  return (terrain_height, land_mask)
  


if __name__ == '__main__':
  argv = sys.argv
  if len(argv) != 2:
      print('Usage: %s <output_array[.npz]>' % (argv[0],))
      sys.exit(-1)

  output_path = argv[1]

  dim = 512
  disc_radius = 1.0
  max_delta = 0.05
  river_downcutting_constant = 1.3
  directional_inertia = 0.4
  default_water_level = 1.0
  evaporation_rate = 0.2
  
  terrain_height, land_mask = get_all(dim, disc_radius, max_delta, river_downcutting_constant, directional_inertia, default_water_level, evaporation_rate)
  np.savez(output_path, height=terrain_height, land_mask=land_mask)





# Various common functions.



# Renormalizes the values of `x` to `bounds`
def normalize(x, bounds=(0, 1)):
  return np.interp(x, (x.min(), x.max()), bounds)


# Fourier-based power law noise with frequency bounds.
def fbm(shape, p, lower=-np.inf, upper=np.inf):
  freqs = tuple(np.fft.fftfreq(n, d=1.0 / n) for n in shape)
  freq_radial = np.hypot(*np.meshgrid(*freqs))
  envelope = (np.power(freq_radial, p, where=freq_radial!=0) *
              (freq_radial > lower) * (freq_radial < upper))
  envelope[0][0] = 0.0
  phase_noise = np.exp(2j * np.pi * np.random.rand(*shape))
  return normalize(np.real(np.fft.ifft2(np.fft.fft2(phase_noise) * envelope)))


# Returns the gradient of the gaussian blur of `a` encoded as a complex number. 
def gaussian_gradient(a, sigma=1.0):
  [fy, fx] = np.meshgrid(*(np.fft.fftfreq(n, 1.0 / n) for n in a.shape))
  sigma2 = sigma**2
  g = lambda x: ((2 * np.pi * sigma2) ** -0.5) * np.exp(-0.5 * (x / sigma)**2)
  dg = lambda x: g(x) * (x / sigma2)

  fa = np.fft.fft2(a)
  dy = np.fft.ifft2(np.fft.fft2(dg(fy) * g(fx)) * fa).real
  dx = np.fft.ifft2(np.fft.fft2(g(fy) * dg(fx)) * fa).real
  return 1j * dx + dy


# Linear interpolation of `x` to `y` with respect to `a`
def lerp(x, y, a): return (1.0 - a) * x + a * y


# Returns a list of grid coordinates for every (x, y) position bounded by
# `shape`
def make_grid_points(shape):
  [Y, X] = np.meshgrid(np.arange(shape[0]), np.arange(shape[1])) 
  grid_points = np.column_stack([X.flatten(), Y.flatten()])
  return grid_points


# Returns a list of points sampled within the bounds of `shape` and with a
# minimum spacing of `radius`.
# NOTE: This function is fairly slow, given that it is implemented with almost
# no array operations.
def poisson_disc_sampling(shape, radius, retries=16):
  grid = {}
  points = []

  # The bounds of `shape` are divided into a grid of cells, each of which can
  # contain a maximum of one point.
  cell_size = radius / np.sqrt(2)
  cells = np.ceil(np.divide(shape, cell_size)).astype(int)
  offsets = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1),
             (1, -1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]
  to_cell = lambda p: (p / cell_size).astype('int')

  # Returns true if there is a point within `radius` of `p`.
  def has_neighbors_in_radius(p):
    cell = to_cell(p)
    for offset in offsets:
      cell_neighbor = (cell[0] + offset[0], cell[1] + offset[1])
      if cell_neighbor in grid:
        p2 = grid[cell_neighbor]
        diff = np.subtract(p2, p)
        if np.dot(diff, diff) <= radius * radius:
          return True
    return False      

  # Adds point `p` to the cell grid.
  def add_point(p):
    grid[tuple(to_cell(p))] = p
    q.append(p)
    points.append(p)

  q = collections.deque()
  first = shape * np.random.rand(2)
  add_point(first)
  while len(q) > 0:
    point = q.pop()

    # Make `retries` attemps to find a point within [radius, 2 * radius] from
    # `point`.
    for _ in range(retries):
      diff = 2 * radius * (2 * np.random.rand(2) - 1)
      r2 = np.dot(diff, diff)
      new_point = diff + point
      if (new_point[0] >= 0 and new_point[0] < shape[0] and
          new_point[1] >= 0 and new_point[1] < shape[1] and 
          not has_neighbors_in_radius(new_point) and
          r2 > radius * radius and r2 < 4 * radius * radius):
        add_point(new_point)
  num_points = len(points)

  # Return points list as a numpy array.
  return np.concatenate(points).reshape((num_points, 2))


# Returns an array in which all True values of `mask` contain the distance to
# the nearest False value.
def dist_to_mask(mask):
  border_mask = (np.maximum.reduce([
      np.roll(mask, 1, axis=0), np.roll(mask, -1, axis=0),
      np.roll(mask, -1, axis=1), np.roll(mask, 1, axis=1)]) * (1 - mask))
  border_points = np.column_stack(np.where(border_mask > 0))

  kdtree = sp.spatial.cKDTree(border_points)
  grid_points = make_grid_points(mask.shape)

  return kdtree.query(grid_points)[0].reshape(mask.shape)


# Peforms a gaussian blur of `a`.
def gaussian_blur(a, sigma=1.0):
  freqs = tuple(np.fft.fftfreq(n, d=1.0 / n) for n in a.shape)
  freq_radial = np.hypot(*np.meshgrid(*freqs))
  sigma2 = sigma**2
  g = lambda x: ((2 * np.pi * sigma2) ** -0.5) * np.exp(-0.5 * (x / sigma)**2)
  kernel = g(freq_radial)
  kernel /= kernel.sum()
  return np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(kernel)).real

