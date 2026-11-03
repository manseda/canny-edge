#include "canny.h"

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <functional>
#include <iomanip>

namespace {

void partitionRows(int rows, int rank, int processCount, int& firstRow, int& lastRow) {
    firstRow = static_cast<int>((static_cast<long long>(rows) * rank) / processCount);
    lastRow = static_cast<int>((static_cast<long long>(rows) * (rank + 1)) / processCount);
}

void copySobelEdges(std::vector<double>& G, std::vector<int>& theta, int sizeRows, int sizeCols) {
    for (int i = 1; i < sizeRows - 1; i++) {
        for (int j = 1; j < sizeCols - 1; j++) {
            if (i == 1) {
                G[i * sizeCols + j - 1] = G[i * sizeCols + j];
                theta[i * sizeCols + j - 1] = theta[i * sizeCols + j];
            } else if (j == 1) {
                G[(i - 1) * sizeCols + j] = G[i * sizeCols + j];
                theta[(i - 1) * sizeCols + j] = theta[i * sizeCols + j];
            } else if (i == sizeRows - 1) {
                G[i * sizeCols + j + 1] = G[i * sizeCols + j];
                theta[i * sizeCols + j + 1] = theta[i * sizeCols + j];
            } else if (j == sizeCols - 1) {
                G[(i + 1) * sizeCols + j] = G[i * sizeCols + j];
                theta[(i + 1) * sizeCols + j] = theta[i * sizeCols + j];
            }

            if (i == 1 && j == 1) {
                G[(i - 1) * sizeCols + j - 1] = G[i * sizeCols + j];
                theta[(i - 1) * sizeCols + j - 1] = theta[i * sizeCols + j];
            } else if (i == 1 && j == sizeCols - 1) {
                G[(i - 1) * sizeCols + j + 1] = G[i * sizeCols + j];
                theta[(i - 1) * sizeCols + j + 1] = theta[i * sizeCols + j];
            } else if (i == sizeRows - 1 && j == 1) {
                G[(i + 1) * sizeCols + j - 1] = G[i * sizeCols + j];
                theta[(i + 1) * sizeCols + j - 1] = theta[i * sizeCols + j];
            } else if (i == sizeRows - 1 && j == sizeCols - 1) {
                G[(i + 1) * sizeCols + j + 1] = G[i * sizeCols + j];
                theta[(i + 1) * sizeCols + j + 1] = theta[i * sizeCols + j];
            }
        }
    }
    for (int i = 1; i < sizeRows - 1; i++) {
        for (int j = 1; j < sizeCols - 1; j++) {
            theta[i * sizeCols + j] = round(theta[i * sizeCols + j] / 45) * 45;
        }
    }
}

template <typename T>
void gatherRows(std::vector<T>& values, int rows, int rowWidth, MPI_Datatype type) {
    int processCount = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    std::vector<int> counts(processCount);
    std::vector<int> displacements(processCount);
    for (int process = 0; process < processCount; ++process) {
        int processFirstRow = 0;
        int processLastRow = 0;
        partitionRows(rows, process, processCount, processFirstRow, processLastRow);
        counts[process] = (processLastRow - processFirstRow) * rowWidth;
        displacements[process] = processFirstRow * rowWidth;
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, type, values.data(), counts.data(), displacements.data(), type, MPI_COMM_WORLD);
}

double timeStage(const std::function<void()>& stage) {
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    stage();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

bool cannyEdgeDetection(const std::string& readLocation, const std::string& writeLocation, double lowerThreshold, double higherThreshold) {
    int rank = 0;
    int processCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    cv::Mat img;
    int sizeRows = 0;
    int sizeCols = 0;
    int sizeDepth = 0;
    int imageStatus = 1;
    std::vector<int> pixels;
    if (rank == 0) {
        if (readLocation == writeLocation) {
            std::cerr << "The read file and save file locations cannot be the same.\n";
            imageStatus = 0;
        } else {
            img = cv::imread(readLocation);
            if (img.empty()) {
                std::cerr << "Could not read the image: " << readLocation << "\n";
                imageStatus = 0;
            } else {
                sizeRows = img.rows;
                sizeCols = img.cols;
                sizeDepth = img.channels();
                const long long pixelCount = static_cast<long long>(sizeRows) * sizeCols * sizeDepth;
                if (pixelCount > INT_MAX) {
                    std::cerr << "Image is too large to distribute with MPI.\n";
                    imageStatus = 0;
                } else {
                    pixels = imgToArray(img, img.data, sizeRows, sizeCols, sizeDepth);
                }
            }
        }
    }
    MPI_Bcast(&imageStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!imageStatus) { return false; }

    int dimensions[3] = {sizeRows, sizeCols, sizeDepth};
    MPI_Bcast(dimensions, 3, MPI_INT, 0, MPI_COMM_WORLD);
    sizeRows = dimensions[0];
    sizeCols = dimensions[1];
    sizeDepth = dimensions[2];
    const int pixelCount = sizeRows * sizeCols * sizeDepth;
    if (rank != 0) { pixels.resize(pixelCount); }
    MPI_Bcast(pixels.data(), pixelCount, MPI_INT, 0, MPI_COMM_WORLD);

    int firstRow = 0;
    int lastRow = 0;
    partitionRows(sizeRows, rank, processCount, firstRow, lastRow);

    // BUFFERS: every buffer is allocated here, outside the timed regions, so the
    // timings only measure the algorithmic cost of each stage.

    std::vector<std::vector<double>> kernel = {{2.0, 4.0, 5.0, 4.0, 2.0},
                                               {4.0, 9.0, 12.0, 9.0, 4.0},
                                               {5.0, 12.0, 15.0, 12.0, 5.0},
                                               {4.0, 9.0, 12.0, 9.0, 4.0},
                                               {2.0, 4.0, 5.0, 4.0, 2.0}};
    double kernelConst = (1.0 / 159.0);
    std::vector<int> pixelsBlur(sizeRows * sizeCols * sizeDepth);
    std::vector<int> pixelsGray(sizeRows * sizeCols);
    std::vector<double> G(sizeRows * sizeCols);
    std::vector<int> theta(sizeRows * sizeCols);
    std::vector<int> pixelsCanny(sizeRows * sizeCols);
    double blurMs, grayscaleMs, sobelMs, nmsMs, thresholdMs;

    // GAUSSIAN_BLUR:
    blurMs = timeStage([&]() {
        gaussianBlur(pixels, pixelsBlur, kernel, kernelConst, sizeRows, sizeCols, sizeDepth, firstRow, lastRow);
        gatherRows(pixelsBlur, sizeRows, sizeCols * sizeDepth, MPI_INT);
    });

    // GRAYSCALE:
    grayscaleMs = timeStage([&]() {
        rgbToGrayscale(pixelsBlur, pixelsGray, sizeRows, sizeCols, sizeDepth, firstRow, lastRow);
        gatherRows(pixelsGray, sizeRows, sizeCols, MPI_INT);
    });

    // SOBEL_GRADIENT:
    double largestG = 0.0;
    sobelMs = timeStage([&]() {
        const double localLargestG = sobelGradient(pixelsGray, G, theta, sizeRows, sizeCols, firstRow, lastRow);
        gatherRows(G, sizeRows, sizeCols, MPI_DOUBLE);
        gatherRows(theta, sizeRows, sizeCols, MPI_INT);
        copySobelEdges(G, theta, sizeRows, sizeCols);
        MPI_Allreduce(&localLargestG, &largestG, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    });

    // NON_MAXIMUM_SUPPRESSION:
    nmsMs = timeStage([&]() {
        nonMaximumSuppression(G, theta, pixelsCanny, largestG, sizeRows, sizeCols, firstRow, lastRow);
        gatherRows(G, sizeRows, sizeCols, MPI_DOUBLE);
        gatherRows(pixelsCanny, sizeRows, sizeCols, MPI_INT);
    });

    // DOUBLE_THRESHOLD (with hysteresis):
    thresholdMs = timeStage([&]() {
        doubleThreshold(G, pixelsCanny, largestG, lowerThreshold, higherThreshold, sizeRows, sizeCols, firstRow, lastRow);
    });

    // Write the result before showing the window: waitKey blocks until a key is
    // pressed, so writing afterwards would hang forever without ever saving.
    int outputStatus = 1;
    if (rank == 0) {
        cv::Mat imgGrayscale(sizeRows, sizeCols, CV_8UC1, cv::Scalar(0));
        arrayToImg(pixelsCanny, imgGrayscale.data, sizeRows, sizeCols, 1);
        if (!cv::imwrite(writeLocation, imgGrayscale)) {
            std::cerr << "Could not write the image: " << writeLocation << "\n";
            outputStatus = 0;
        } else {
            std::cout << "Saved the result to: " << writeLocation << "\n";

            // The preview window is optional so the program can also run unattended.
            if (std::getenv("CANNY_NO_PREVIEW") == nullptr) {
                try {
                    cv::imshow("CannyEdgeDetection", imgGrayscale);
                    cv::waitKey(0);
                    cv::destroyAllWindows();
                } catch (const cv::Exception& e) {
                    std::cerr << "Preview window unavailable: " << e.what() << "\n";
                }
            }
        }
    }
    MPI_Bcast(&outputStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!outputStatus) { return false; }

    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(3)
                  << "\nTiming (" << sizeCols << "x" << sizeRows << ", " << processCount << " MPI processes, lower=" << lowerThreshold << ", higher=" << higherThreshold << "):\n"
                  << "  gaussian blur           : " << blurMs << " ms\n"
                  << "  grayscale               : " << grayscaleMs << " ms\n"
                  << "  sobel gradient          : " << sobelMs << " ms\n"
                  << "  non-maximum suppression : " << nmsMs << " ms\n"
                  << "  double threshold        : " << thresholdMs << " ms\n"
                  << "  total (5 stages)        : " << (blurMs + grayscaleMs + sobelMs + nmsMs + thresholdMs) << " ms\n";
    }

    return true;
}

std::vector<int> imgToArray(cv::Mat img, uint8_t* pixelPtr, int sizeRows, int sizeCols, int sizeDepth) {
    std::vector<int> pixels(sizeRows * sizeCols * sizeDepth);
    for (int i = 0; i < sizeRows; i++) {
        for (int j = 0; j < sizeCols; j++) {
            for (int k = 0; k < sizeDepth; k++) {
                // converting BGR to RGB colors
                pixels[i * sizeCols * sizeDepth + j * sizeDepth + k] =
                    (int)pixelPtr[i * sizeCols * sizeDepth + j * sizeDepth + 2 - k];
            }
        }
    }
    return pixels;
}

void arrayToImg(std::vector<int>& pixels, uint8_t* pixelPtr, int sizeRows, int sizeCols, int sizeDepth) {
    for (int i = 0; i < sizeRows; i++) {
        for (int j = 0; j < sizeCols; j++) {
            for (int k = 0; k < sizeDepth; k++) {
                pixelPtr[i * sizeCols * sizeDepth + j * sizeDepth + k] =
                    (uint8_t)pixels[i * sizeCols * sizeDepth + j * sizeDepth + (sizeDepth - 1 - k)];
            }
        }
    }
    return;
}

void gaussianBlur(const std::vector<int>& pixels, std::vector<int>& pixelsBlur, const std::vector<std::vector<double>>& kernel, double kernelConst, int sizeRows, int sizeCols, int sizeDepth, int firstRow, int lastRow) {
    if (lastRow < 0) { lastRow = sizeRows; }
    for (int i = firstRow; i < lastRow; i++) {
        for (int j = 0; j < sizeCols; j++) {
            for (int k = 0; k < sizeDepth; k++) {
                double sum = 0;
                double sumKernel = 0;
                for (int y = -2; y <= 2; y++) {
                    for (int x = -2; x <= 2; x++) {
                        if ((i + x) >= 0 && (i + x) < sizeRows && (j + y) >= 0 && (j + y) < sizeCols) {
                            double channel = (double)pixels[(i + x) * sizeCols * sizeDepth + (j + y) * sizeDepth + k];
                            sum += channel * kernelConst * kernel[x + 2][y + 2];
                            sumKernel += kernelConst * kernel[x + 2][y + 2];
                        }
                    }
                }
                pixelsBlur[i * sizeCols * sizeDepth + j * sizeDepth + k] = (int)(sum / sumKernel);
            }
        }
    }
}

void rgbToGrayscale(const std::vector<int>& pixels, std::vector<int>& pixelsGray, int sizeRows, int sizeCols, int sizeDepth, int firstRow, int lastRow) {
    if (lastRow < 0) { lastRow = sizeRows; }
    for (int i = firstRow; i < lastRow; i++) {
        for (int j = 0; j < sizeCols; j++) {
            int sum = 0;
            for (int k = 0; k < sizeDepth; k++) {
                sum = sum + pixels[i * sizeCols * sizeDepth + j * sizeDepth + k];
            }
            pixelsGray[i * sizeCols + j] = (int)(sum / sizeDepth);
        }
    }
}

// Sobel gradient: fills the magnitude G and the direction theta (rounded to the
// nearest 45 degrees) and returns the largest magnitude found.
double sobelGradient(const std::vector<int>& pixels, std::vector<double>& G, std::vector<int>& theta, int sizeRows, int sizeCols, int firstRow, int lastRow) {
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    double largestG = 0;

    if (lastRow < 0) { lastRow = sizeRows; }
    // The one-row halo is already available because the preceding stage is gathered.
    for (int i = std::max(firstRow, 1); i < std::min(lastRow, sizeRows - 1); i++) {
        for (int j = 1; j < sizeCols - 1; j++) {
            // find gx and gy for each pixel
            double gxValue = 0;
            double gyValue = 0;
            for (int x = -1; x <= 1; x++) {
                for (int y = -1; y <= 1; y++) {
                    gxValue = gxValue + (gx[1 - x][1 - y] * (double)(pixels[(i + x) * sizeCols + j + y]));
                    gyValue = gyValue + (gy[1 - x][1 - y] * (double)(pixels[(i + x) * sizeCols + j + y]));
                }
            }

            // calculate G and theta
            G[i * sizeCols + j] = std::sqrt(std::pow(gxValue, 2) + std::pow(gyValue, 2));
            double atanResult = atan2(gyValue, gxValue) * 180.0 / 3.14159265;
            theta[i * sizeCols + j] = (int)(180.0 + atanResult);

            if (G[i * sizeCols + j] > largestG) { largestG = G[i * sizeCols + j]; }

        }
    }
    return largestG;
}

void nonMaximumSuppression(std::vector<double>& G, const std::vector<int>& theta, std::vector<int>& pixelsCanny, double largestG, int sizeRows, int sizeCols, int firstRow, int lastRow) {
    if (lastRow < 0) { lastRow = sizeRows; }
    const std::vector<double> gradients = G;
    for (int i = std::max(firstRow, 1); i < std::min(lastRow, sizeRows - 1); i++) {
        for (int j = 1; j < sizeCols - 1; j++) {
            if (theta[i * sizeCols + j] == 0 || theta[i * sizeCols + j] == 180) {
                if (gradients[i * sizeCols + j] < gradients[i * sizeCols + j - 1] || gradients[i * sizeCols + j] < gradients[i * sizeCols + j + 1]) {
                    G[i * sizeCols + j] = 0;
                }
            } else if (theta[i * sizeCols + j] == 45 || theta[i * sizeCols + j] == 225) {
                if (gradients[i * sizeCols + j] < gradients[(i + 1) * sizeCols + j + 1] || gradients[i * sizeCols + j] < gradients[(i - 1) * sizeCols + j - 1]) {
                    G[i * sizeCols + j] = 0;
                }
            } else if (theta[i * sizeCols + j] == 90 || theta[i * sizeCols + j] == 270) {
                if (gradients[i * sizeCols + j] < gradients[(i + 1) * sizeCols + j] || gradients[i * sizeCols + j] < gradients[(i - 1) * sizeCols + j]) {
                    G[i * sizeCols + j] = 0;
                }
            } else {
                if (gradients[i * sizeCols + j] < gradients[(i + 1) * sizeCols + j - 1] || gradients[i * sizeCols + j] < gradients[(i - 1) * sizeCols + j + 1]) {
                    G[i * sizeCols + j] = 0;
                }
            }

            pixelsCanny[i * sizeCols + j] = largestG > 0.0 ? (int)(G[i * sizeCols + j] * (255.0 / largestG)) : 0;
        }
    }
}

void doubleThreshold(std::vector<double>& G, std::vector<int>& pixelsCanny, double largestG, double lowerThreshold, double higherThreshold, int sizeRows, int sizeCols, int firstRow, int lastRow) {
    if (lastRow < 0) { lastRow = sizeRows; }
    const double low = lowerThreshold * largestG;
    const double high = higherThreshold * largestG;
    const std::vector<double> candidates = G;
    bool globalChanges = false;
    do {
        std::vector<double> nextG = G;
        int localChanges = 0;
        if (largestG > 0.0) {
            for (int i = std::max(firstRow, 1); i < std::min(lastRow, sizeRows - 1); i++) {
                for (int j = 1; j < sizeCols - 1; j++) {
                    const int index = i * sizeCols + j;
                    if (candidates[index] < low || candidates[index] >= high || G[index] >= high) {
                        continue;
                    }

                    bool connectedToStrongEdge = false;
                    for (int x = -1; x <= 1 && !connectedToStrongEdge; ++x) {
                        for (int y = -1; y <= 1; ++y) {
                            if ((x != 0 || y != 0) && G[(i + x) * sizeCols + j + y] >= high) {
                                connectedToStrongEdge = true;
                                break;
                            }
                        }
                    }
                    if (connectedToStrongEdge) {
                        nextG[index] = high;
                        localChanges = 1;
                    }
                }
            }
        }

        gatherRows(nextG, sizeRows, sizeCols, MPI_DOUBLE);
        G.swap(nextG);
        MPI_Allreduce(&localChanges, &globalChanges, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    } while (globalChanges);

    for (int i = std::max(firstRow, 1); i < std::min(lastRow, sizeRows - 1); i++) {
        for (int j = 0; j < sizeCols; j++) {
            const int index = i * sizeCols + j;
            if (candidates[index] < low || (candidates[index] < high && G[index] < high)) { G[index] = 0.0; }
            pixelsCanny[index] = largestG > 0.0 ? static_cast<int>(G[index] * (255.0 / largestG)) : 0;
        }
    }
    gatherRows(G, sizeRows, sizeCols, MPI_DOUBLE);
    gatherRows(pixelsCanny, sizeRows, MPI_INT);
}
