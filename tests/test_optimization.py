"""pygeoprocessing.raster_optimization test suite."""
import os
import shutil
import tempfile
import unittest
import time

from osgeo import gdal
from osgeo import osr
import numpy

import pygeoprocessing
import pygeoprocessing.routing


class TestOptimization(unittest.TestCase):
    """Tests for pygeoprocessing.raster_optimization."""
    def setUp(self):
        """Create a temporary workspace that's deleted later."""
        self.workspace_dir = tempfile.mkdtemp()
        if not os.path.exists(self.workspace_dir):
            os.makedirs(self.workspace_dir)

    def tearDown(self):
        """Clean up remaining files."""
        shutil.rmtree(self.workspace_dir)

    def test_optimization(self):
        """PGP.raster_optimization: test."""
        n = 1000
        target_suffix = 'normal_100'
        test_array = numpy.random.normal(scale=100, size=(n, n))
        # target_suffix = 'random'
        # test_array = numpy.random.random((n, n))
        # target_suffix = 'inv_normal_100'
        # test_array = 1 / numpy.random.normal(scale=100, size=(n, n))
        workspace_dir = 'optimization_test'
        test_raster_path = os.path.join(workspace_dir, 'base.tif')
        try:
            os.makedirs(os.path.dirname(test_raster_path))
        except OSError:
            pass

        print('create test raster')
        pygeoprocessing.numpy_array_to_raster(
            test_array, None, (1, -1), (0, 0), None, test_raster_path)

        target_proportion_list = [float(n/100) for n in range(1, 100, 5)] + [.9999]

        proportion_raster_list = []
        sorted_array = numpy.sort(test_array.flatten())
        for target_proportion in target_proportion_list:
            target_sum = numpy.sum(sorted_array) * target_proportion
            running_sum = 0.0
            threshold_val = 0.0
            for index, val in enumerate(sorted_array[::-1]):
                running_sum += val
                if running_sum > target_sum:
                    threshold_val = val
                    break
            expected_result = test_array >= threshold_val
            expected_raster_path = os.path.join(
                workspace_dir,
                f'expected_result{target_proportion}_{target_suffix}.tif')
            pygeoprocessing.numpy_array_to_raster(
                expected_result, None, (1, -1), (0, 0), None,
                expected_raster_path)
            proportion_raster_list.append(
                (target_proportion, expected_raster_path))

        # need to test the isclose because it's float and doesn't quite equal
        # the numpy floating point standard
        churn_directory = os.path.join(workspace_dir, 'churn')
        pygeoprocessing.raster_optimization(
            [(test_raster_path, 1)], churn_directory, workspace_dir,
            goal_met_cutoffs=target_proportion_list,
            target_suffix=target_suffix)
        time.sleep(5)
        for target_proportion, expected_raster_path in proportion_raster_list:
            result_raster_path = os.path.join(
                workspace_dir, f'optimal_mask_{target_proportion}_{target_suffix}.tif')
            print(result_raster_path)
            result_array = pygeoprocessing.raster_to_numpy_array(
                result_raster_path)
            expected_result = pygeoprocessing.raster_to_numpy_array(
                expected_raster_path)
            numpy.testing.assert_array_equal(expected_result, result_array)
