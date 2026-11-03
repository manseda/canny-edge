# canny-edge-detector

A Canny edge detector written in C++. OpenCV is used only for image reading/writing, and MPI distributes the image processing across processes. The detector implements a Gaussian blur, a Sobel filter as the edge detection operator, non-maximum suppression, and double thresholding.

## Parallel execution with MPI

The image processing stages run on row partitions distributed across MPI
processes. The root process reads the input image, gathers the result, and
writes the output.

Configure and build with CMake, OpenCV, and an MPI implementation that provides
the C++ bindings:

```sh
cmake -S . -B build
cmake --build build --config Release
```

Run the executable through the MPI launcher:

```sh
mpiexec -n 4 ./build/canny images/Sukuna.jpg
```

The existing output and threshold arguments are supported:

```sh
mpiexec -n 4 ./build/canny images/Sukuna.jpg result.jpg 0.03 0.1
```

On Windows, use the executable produced by the selected CMake generator (for
example `build\Release\canny.exe`) and the MPI launcher installed with the MPI
implementation.

## Example:

### Original Image

<img src="images/Sukuna.jpg" alt="Original Image">

### Image after Canny Edge Detection

<img src="images/SukunaCanny.jpg" alt="Image after Canny Edge Detection">
