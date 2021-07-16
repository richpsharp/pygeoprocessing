# coding=UTF-8
"""Optimization routines for raster calculations."""


LOGGER = logging.getLogger('pygeoprocessing.optimization')


cdef extern from "FastFileIterator.h" nogil:
    cdef cppclass FastFileIterator[DATA_T]:
        FastFileIterator(const char*, size_t)
        DATA_T next()
        size_t size()
    int FastFileIteratorCompare[DATA_T](FastFileIterator[DATA_T]*,
                                        FastFileIterator[DATA_T]*)

cdef extern from "<algorithm>" namespace "std":
    void push_heap(...)
    void pop_heap(...)


def greedy_pixel_pick_by_area(
        base_value_raster_path_band, area_per_pixel_raster_path_band,
        selected_area_report_list, output_dir, output_prefix=None):
    """Select pixels from base raster from largest to smallest.

    Creates a set of stepwise raster masks of pixel selection based on
    selecting the highest value pixels before lower value pixels (greedy).
    A series of rasters and a table are written to ``output_dir``L
        * Rasters filename form "[output_prefix]step_[area].tif" where "area"
          is the amount of area in units of the
          ``area_per_pixel_raster_path_band`` raster.
        * A csv table named "[output_prefix]greedy_pixel_pick_result.csv" that
          contains two columns "area" and "total_value" where "area"
          corresponds to the area step in the previous raster and "total value"
          corresponds to the sum of the base raster pixels selected so far.

    Args:
        base_value_raster_path_band (tuple): the raster that is used for
            greedy order selection and the values as the sum selected
        area_per_pixel_raster_path_band (tuple): a raster of the same size
            and projection as ``base_value_raster_path_band`` whose pixels
            correspond to the area of the pixels in whatever units the caller
            desires.
        selected_area_report_list (list): a list of increasing areas in the
            same units as ``area_per_pixel_raster_path_band`` to report
            total value selected, to write a raster mask, and create a row in
            "[output_prefix]greedy_pixel_pick_result.csv".
        output_dir (str): path to an output directory that will contain
            rasters of the form "[output_prefix]step_[area].tif" and
            an output table named
            "[output_prefix]greedy_pixel_pick_result.csv".
        output_prefix (str): optional parameter to prefix the raster and table
            filename results. If set to None then no prefix is applied.

    Return:
        ``None```
    """
    cdef FILE *fptr
    cdef double[:] buffer_data
    cdef FastFileIteratorDoublePtr fast_file_iterator
    cdef vector[FastFileIteratorDoublePtr] fast_file_iterator_vector
    cdef int percentile_index = 0
    cdef long long i, n_elements = 0
    cdef double next_val = 0.0
    cdef double current_step = 0.0
    cdef double step_size, current_percentile
    result_list = []
    rm_dir_when_done = False
    try:
        os.makedirs(working_sort_directory)
        rm_dir_when_done = True
    except OSError as e:
        LOGGER.warning("couldn't make working_sort_directory: %s", str(e))
    file_index = 0
    nodata = pygeoprocessing.get_raster_info(
        base_raster_path_band[0])['nodata'][base_raster_path_band[1]-1]
    heapfile_list = []

    raster_info = pygeoprocessing.get_raster_info(
        base_raster_path_band[0])
    nodata = raster_info['nodata'][base_raster_path_band[1]-1]
    cdef long long n_pixels = (
        raster_info['raster_size'][0] * raster_info['raster_size'][1])
    cdef long long pixels_processed = 0

    last_update = time.time()
    LOGGER.debug('sorting data to heap')
    for _, block_data in pygeoprocessing.iterblocks(
            base_raster_path_band, largest_block=heap_buffer_size):
        pixels_processed += block_data.size
        if time.time() - last_update > 5.0:
            LOGGER.debug(
                f'data sort to heap {(100.*pixels_processed)/n_pixels:.1f}% '
                f'complete, {pixels_processed} out of {n_pixels}'),

            last_update = time.time()
        if nodata is not None:
            clean_data = block_data[~numpy.isclose(block_data, nodata)]
        else:
            clean_data = block_data.flatten()
        clean_data = clean_data[numpy.isfinite(clean_data)]
        buffer_data = numpy.sort(clean_data).astype(numpy.double)
        if buffer_data.size == 0:
            continue
        n_elements += buffer_data.size
        file_path = os.path.join(
            working_sort_directory, '%d.dat' % file_index)
        heapfile_list.append(file_path)
        fptr = fopen(bytes(file_path.encode()), "wb")
        fwrite(
            <double*>&buffer_data[0], sizeof(double), buffer_data.size, fptr)
        fclose(fptr)
        file_index += 1

        fast_file_iterator = new FastFileIterator[double](
            (bytes(file_path.encode())), ffi_buffer_size)
        fast_file_iterator_vector.push_back(fast_file_iterator)
        push_heap(
            fast_file_iterator_vector.begin(),
            fast_file_iterator_vector.end(),
            FastFileIteratorCompare[double])

    current_percentile = percentile_list[percentile_index]
    step_size = 0
    if n_elements > 0:
        step_size = 100.0 / n_elements

    LOGGER.debug('calculating percentiles')
    for i in range(n_elements):
        if time.time() - last_update > 5.0:
            LOGGER.debug(
                'calculating percentiles %.2f%% complete',
                100.0 * i / float(n_elements))
            last_update = time.time()
        current_step = step_size * i
        next_val = fast_file_iterator_vector.front().next()
        if current_step >= current_percentile:
            result_list.append(next_val)
            percentile_index += 1
            if percentile_index >= len(percentile_list):
                break
            current_percentile = percentile_list[percentile_index]
        pop_heap(
            fast_file_iterator_vector.begin(),
            fast_file_iterator_vector.end(),
            FastFileIteratorCompare[double])
        if fast_file_iterator_vector.back().size() > 0:
            push_heap(
                fast_file_iterator_vector.begin(),
                fast_file_iterator_vector.end(),
                FastFileIteratorCompare[double])
        else:
            fast_file_iterator_vector.pop_back()
    if percentile_index < len(percentile_list):
        result_list.append(next_val)
    # free all the iterator memory
    ffiv_iter = fast_file_iterator_vector.begin()
    while ffiv_iter != fast_file_iterator_vector.end():
        fast_file_iterator = deref(ffiv_iter)
        del fast_file_iterator
        inc(ffiv_iter)
    fast_file_iterator_vector.clear()
    # delete all the heap files
    for file_path in heapfile_list:
        try:
            os.remove(file_path)
        except OSError:
            # you never know if this might fail!
            LOGGER.warning('unable to remove %s', file_path)
    if rm_dir_when_done:
        shutil.rmtree(working_sort_directory)
    LOGGER.debug('here is percentile_list: %s', str(result_list))
    return result_list
