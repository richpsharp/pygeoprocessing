"""pygeoprocessing.optimization test suite."""
import itertools
import os
import pathlib
import shutil
import tempfile
import time
import types
import unittest
import unittest.mock

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from numpy.random import MT19937
from numpy.random import RandomState
from numpy.random import SeedSequence
import numpy
import scipy.ndimage
import shapely.geometry
import shapely.wkt

import pygeoprocessing
import pygeoprocessing.multiprocessing
import pygeoprocessing.symbolic
from pygeoprocessing.geoprocessing_core import \
    DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS

_DEFAULT_ORIGIN = (444720, 3751320)
_DEFAULT_PIXEL_SIZE = (30, -30)
_DEFAULT_EPSG = 3116


def _array_to_raster(
        base_array, target_nodata, target_path,
        creation_options=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS[1],
        pixel_size=_DEFAULT_PIXEL_SIZE, projection_epsg=_DEFAULT_EPSG,
        origin=_DEFAULT_ORIGIN):
    """Passthrough to pygeoprocessing.array_to_raster."""
    projection = osr.SpatialReference()
    projection_wkt = None
    if projection_epsg is not None:
        projection.ImportFromEPSG(projection_epsg)
        projection_wkt = projection.ExportToWkt()
    pygeoprocessing.numpy_array_to_raster(
        base_array, target_nodata, pixel_size, origin, projection_wkt,
        target_path, raster_driver_creation_tuple=('GTiff', creation_options))


class TestOptimization(unittest.TestCase):
    """Tests for pygeoprocessing.optimization."""

    def setUp(self):
        """Create a temporary workspace that's deleted later."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up remaining files."""
        shutil.rmtree(self.workspace_dir)

    def test_greedy_pixel_pick(self):
        """PGP.optimization: test greedy pixel pick."""
        n_pixels = 9
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        pixel_matrix[:] = test_value
        pixel_matrix[-1, 0] = test_value - 1  # making a bad value
        target_nodata = -1
        value_path = os.path.join(self.workspace_dir, 'raster.tif')
        _array_to_raster(
            pixel_matrix, target_nodata, value_path)

        area_path = os.path.join(self.workspace_dir, 'area_raster.tif')
        _array_to_raster(
            pixel_matrix, target_nodata, area_path)

        pygeoprocessing.optimization.greedy_pixel_pick_by_area(
            (value_path, 1), (area_path, 1),
            [1, 10, 100], self.workspace_dir, output_prefix='test_')

        self.assertTrue(os.path.exists(
            os.path.join(self.workspace_dir, 'test_result.csv')))
