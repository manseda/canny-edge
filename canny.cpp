#include "canny.h"

#include <chrono>
#include <iomanip>

// TODO: implement a function to read and write the images.

namespace {

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

}  // namespace

bool cannyEdgeDetection(const std::string& readLocation, const std::string& writeLocation, double lowerThreshold, double higherThreshold) {
    const Clock::time_point start = Clock::now();

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
    const double readMs = elapsedMs(start);

    // GAUSSIAN_BLUR:

    std::vector<std::vector<double>> kernel = {{2.0, 4.0, 5.0, 4.0, 2.0},
                                               {4.0, 9.0, 12.0, 9.0, 4.0},
                                               {5.0, 12.0, 15.0, 12.0, 5.0},
                                               {4.0, 9.0, 12.0, 9.0, 4.0},
                                               {2.0, 4.0, 5.0, 4.0, 2.0}};
    double kernelConst = (1.0 / 159.0);
    std::vector<int> pixelsBlur = gaussianBlur(pixels, kernel, kernelConst, sizeRows, sizeCols, sizeDepth);

    arrayToImg(pixelsBlur, pixelPtr, sizeRows, sizeCols, sizeDepth);
    // cv::imshow("Blurred", img);
    const double blurMs = elapsedMs(start);

    // GRAYSCALE:

    cv::Mat imgGrayscale(sizeRows, sizeCols, CV_8UC1, cv::Scalar(0));
    uint8_t* pixelPtrGray = (uint8_t*)imgGrayscale.data;

    std::vector<int> pixelsGray = rgbToGrayscale(pixelsBlur, sizeRows, sizeCols, sizeDepth);
    arrayToImg(pixelsGray, pixelPtrGray, sizeRows, sizeCols, 1);
    // cv::imshow("Grayscale", imgGrayscale);
    const double grayscaleMs = elapsedMs(start);

    // CANNY_FILTER:

    std::vector<int> pixelsCanny = cannyFilter(pixelsGray, sizeRows, sizeCols, 1, lowerThreshold, higherThreshold);
    arrayToImg(pixelsCanny, pixelPtrGray, sizeRows, sizeCols, 1);
    const double cannyMs = elapsedMs(start);

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

    std::cout << std::fixed << std::setprecision(2)
              << "\nTiming (" << sizeCols << "x" << sizeRows << ", lower=" << lowerThreshold << ", higher=" << higherThreshold << "):\n"
              << "  read + pixel conversion : " << readMs << " ms\n"
              << "  gaussian blur          : " << (blurMs - readMs) << " ms\n"
              << "  grayscale              : " << (grayscaleMs - blurMs) << " ms\n"
              << "  canny (sobel+NMS+thres): " << (cannyMs - grayscaleMs) << " ms\n"
              << "  total (excl. saving)   : " << cannyMs << " ms\n";

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

std::vector<int> gaussianBlur(std::vector<int>& pixels, std::vector<std::vector<double>>& kernel, double kernelConst, int sizeRows, int sizeCols, int sizeDepth) {
    std::vector<int> pixelsBlur(sizeRows * sizeCols * sizeDepth);
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
    return pixelsBlur;
}

std::vector<int> rgbToGrayscale(std::vector<int>& pixels, int sizeRows, int sizeCols, int sizeDepth) {
    std::vector<int> pixelsGray(sizeRows * sizeCols);
    for (int i = 0; i < sizeRows; i++) {
        for (int j = 0; j < sizeCols; j++) {
            int sum = 0;
            for (int k = 0; k < sizeDepth; k++) {
                sum = sum + pixels[i * sizeCols * sizeDepth + j * sizeDepth + k];
            }
            pixelsGray[i * sizeCols + j] = (int)(sum / sizeDepth);
        }
    }
    return pixelsGray;
}

std::vector<int> cannyFilter(std::vector<int>& pixels, int sizeRows, int sizeCols, int sizeDepth, double lowerThreshold, double higherThreshold) {
    std::vector<int> pixelsCanny(sizeRows * sizeCols);
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    double* G = new double[sizeRows * sizeCols];
    std::vector<int> theta(sizeRows * sizeCols);
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

    // non-maximum suppression
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

    // double threshold
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

    return pixelsCanny;
}
