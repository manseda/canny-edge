#include "canny.h"

#include <time.h>

#include <iomanip>

// TODO: implement a function to read and write the images.

namespace {

// Milliseconds between two CLOCK_MONOTONIC samples.
double elapsedMs(const struct timespec& start, const struct timespec& end) {
    return (double)(end.tv_sec - start.tv_sec) * 1000.0 + (double)(end.tv_nsec - start.tv_nsec) / 1.0e6;
}

}  // namespace

bool cannyEdgeDetection(const std::string& readLocation, const std::string& writeLocation, double lowerThreshold, double higherThreshold) {
    if (readLocation == writeLocation) {
        std::cerr << "The read file and save file locations cannot be the same.\n";
        return false;
    }
    cv::Mat img = cv::imread(readLocation);
    if (img.empty()) {
        std::cerr << "Could not read the image: " << readLocation << "\n";
        return false;
    }

    // READ_FILE:

    uint8_t* pixelPtr = (uint8_t*)img.data;
    int sizeRows = img.rows;
    int sizeCols = img.cols;
    int sizeDepth = img.channels();
    std::vector<int> pixels = imgToArray(img, pixelPtr, sizeRows, sizeCols, sizeDepth);

    arrayToImg(pixels, pixelPtr, sizeRows, sizeCols, sizeDepth);
    // cv::imshow("Original", img);

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
    cv::Mat imgGrayscale(sizeRows, sizeCols, CV_8UC1, cv::Scalar(0));
    uint8_t* pixelPtrGray = (uint8_t*)imgGrayscale.data;

    struct timespec start, end;
    double blurMs, grayscaleMs, sobelMs, nmsMs, thresholdMs;

    // GAUSSIAN_BLUR:

    clock_gettime(CLOCK_MONOTONIC, &start);
    gaussianBlur(pixels, pixelsBlur, kernel, kernelConst, sizeRows, sizeCols, sizeDepth);
    clock_gettime(CLOCK_MONOTONIC, &end);
    blurMs = elapsedMs(start, end);

    arrayToImg(pixelsBlur, pixelPtr, sizeRows, sizeCols, sizeDepth);
    // cv::imshow("Blurred", img);

    // GRAYSCALE:

    clock_gettime(CLOCK_MONOTONIC, &start);
    rgbToGrayscale(pixelsBlur, pixelsGray, sizeRows, sizeCols, sizeDepth);
    clock_gettime(CLOCK_MONOTONIC, &end);
    grayscaleMs = elapsedMs(start, end);

    arrayToImg(pixelsGray, pixelPtrGray, sizeRows, sizeCols, 1);
    // cv::imshow("Grayscale", imgGrayscale);

    // SOBEL_GRADIENT:

    clock_gettime(CLOCK_MONOTONIC, &start);
    double largestG = sobelGradient(pixelsGray, G, theta, sizeRows, sizeCols);
    clock_gettime(CLOCK_MONOTONIC, &end);
    sobelMs = elapsedMs(start, end);

    // NON_MAXIMUM_SUPPRESSION:

    clock_gettime(CLOCK_MONOTONIC, &start);
    nonMaximumSuppression(G, theta, pixelsCanny, largestG, sizeRows, sizeCols);
    clock_gettime(CLOCK_MONOTONIC, &end);
    nmsMs = elapsedMs(start, end);

    // DOUBLE_THRESHOLD (with hysteresis):

    clock_gettime(CLOCK_MONOTONIC, &start);
    doubleThreshold(G, pixelsCanny, largestG, lowerThreshold, higherThreshold, sizeRows, sizeCols);
    clock_gettime(CLOCK_MONOTONIC, &end);
    thresholdMs = elapsedMs(start, end);

    arrayToImg(pixelsCanny, pixelPtrGray, sizeRows, sizeCols, 1);

    // Write the result before showing the window: waitKey blocks until a key is
    // pressed, so writing afterwards would hang forever without ever saving.
    if (!cv::imwrite(writeLocation, imgGrayscale)) {
        std::cerr << "Could not write the image: " << writeLocation << "\n";
        return false;
    }
    std::cout << "Saved the result to: " << writeLocation << "\n";

    // The preview window is optional so the program can also run unattended (a
    // headless machine, a script, CI). Set CANNY_NO_PREVIEW=1 to skip it.
    if (std::getenv("CANNY_NO_PREVIEW") == nullptr) {
        try {
            cv::imshow("CannyEdgeDetection", imgGrayscale);
            cv::waitKey(0);
            cv::destroyAllWindows();
        } catch (const cv::Exception& e) {
            std::cerr << "Preview window unavailable: " << e.what() << "\n";
        }
    }

    std::cout << std::fixed << std::setprecision(3)
              << "\nTiming (" << sizeCols << "x" << sizeRows << ", lower=" << lowerThreshold << ", higher=" << higherThreshold << "):\n"
              << "  gaussian blur           : " << blurMs << " ms\n"
              << "  grayscale               : " << grayscaleMs << " ms\n"
              << "  sobel gradient          : " << sobelMs << " ms\n"
              << "  non-maximum suppression : " << nmsMs << " ms\n"
              << "  double threshold        : " << thresholdMs << " ms\n"
              << "  total (5 stages)        : " << (blurMs + grayscaleMs + sobelMs + nmsMs + thresholdMs) << " ms\n";

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

void gaussianBlur(const std::vector<int>& pixels, std::vector<int>& pixelsBlur, const std::vector<std::vector<double>>& kernel, double kernelConst, int sizeRows, int sizeCols, int sizeDepth) {
    for (int i = 0; i < sizeRows; i++) {
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

void rgbToGrayscale(const std::vector<int>& pixels, std::vector<int>& pixelsGray, int sizeRows, int sizeCols, int sizeDepth) {
    for (int i = 0; i < sizeRows; i++) {
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
double sobelGradient(const std::vector<int>& pixels, std::vector<double>& G, std::vector<int>& theta, int sizeRows, int sizeCols) {
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    double largestG = 0;

    // perform canny edge detection on everything but the edges
    for (int i = 1; i < sizeRows - 1; i++) {
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

            // setting the edges
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

            // setting the corners
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

            // round to the nearest 45 degrees
            theta[i * sizeCols + j] = round(theta[i * sizeCols + j] / 45) * 45;
        }
    }
    for (int i = 1; i < sizeRows - 1; i++) {
        for (int j = 1; j < sizeCols - 1; j++) {
        }
    }
    return largestG;
}

void nonMaximumSuppression(std::vector<double>& G, const std::vector<int>& theta, std::vector<int>& pixelsCanny, double largestG, int sizeRows, int sizeCols) {
    for (int i = 1; i < sizeRows - 1; i++) {
        for (int j = 1; j < sizeCols - 1; j++) {
            if (theta[i * sizeCols + j] == 0 || theta[i * sizeCols + j] == 180) {
                if (G[i * sizeCols + j] < G[i * sizeCols + j - 1] || G[i * sizeCols + j] < G[i * sizeCols + j + 1]) {
                    G[i * sizeCols + j] = 0;
                }
            } else if (theta[i * sizeCols + j] == 45 || theta[i * sizeCols + j] == 225) {
                if (G[i * sizeCols + j] < G[(i + 1) * sizeCols + j + 1] || G[i * sizeCols + j] < G[(i - 1) * sizeCols + j - 1]) {
                    G[i * sizeCols + j] = 0;
                }
            } else if (theta[i * sizeCols + j] == 90 || theta[i * sizeCols + j] == 270) {
                if (G[i * sizeCols + j] < G[(i + 1) * sizeCols + j] || G[i * sizeCols + j] < G[(i - 1) * sizeCols + j]) {
                    G[i * sizeCols + j] = 0;
                }
            } else {
                if (G[i * sizeCols + j] < G[(i + 1) * sizeCols + j - 1] || G[i * sizeCols + j] < G[(i - 1) * sizeCols + j + 1]) {
                    G[i * sizeCols + j] = 0;
                }
            }

            pixelsCanny[i * sizeCols + j] = (int)(G[i * sizeCols + j] * (255.0 / largestG));
        }
    }
}

void doubleThreshold(std::vector<double>& G, std::vector<int>& pixelsCanny, double largestG, double lowerThreshold, double higherThreshold, int sizeRows, int sizeCols) {
    bool changes;
    do {
        changes = false;
        for (int i = 1; i < sizeRows - 1; i++) {
            for (int j = 1; j < sizeCols - 1; j++) {
                if (G[i * sizeCols + j] < (lowerThreshold * largestG)) {
                    G[i * sizeCols + j] = 0;
                } else if (G[i * sizeCols + j] >= (higherThreshold * largestG)) {
                    continue;
                } else if (G[i * sizeCols + j] < (higherThreshold * largestG)) {
                    int tempG = G[i * sizeCols + j];
                    G[i * sizeCols + j] = 0;
                    for (int x = -1; x <= 1; x++) {
                        bool breakNestedLoop = false;
                        for (int y = -1; y <= 1; y++) {
                            if (x == 0 && y == 0) { continue; }
                            if (G[(i + x) * sizeCols + (j + y)] >= (higherThreshold * largestG)) {
                                G[i * sizeCols + j] = (higherThreshold * largestG);
                                changes = true;
                                breakNestedLoop = true;
                                break;
                            }
                        }
                        if (breakNestedLoop) { break; }
                    }
                }
                pixelsCanny[i * sizeCols + j] = (int)(G[i * sizeCols + j] * (255.0 / largestG));
            }
        }
    } while (changes);
}
