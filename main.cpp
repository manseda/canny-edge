#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

#include "canny.h"

namespace fs = std::filesystem;

namespace {

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [input] [output] [lowerThreshold] [higherThreshold]\n\n"
              << "  input           image to process (default: ../images/Sukuna.jpg, falling back\n"
              << "                  to images/Sukuna.jpg so the program works from the project\n"
              << "                  root or from the build/ directory)\n"
              << "  output          where to write the result (default: <input> with 'Canny'\n"
              << "                  appended to the file name)\n"
              << "  lowerThreshold  low hysteresis threshold, as a fraction of the maximum\n"
              << "                  gradient (default: 0.03)\n"
              << "  higherThreshold high hysteresis threshold, as a fraction of the maximum\n"
              << "                  gradient (default: 0.1)\n\n"
              << "Set CANNY_NO_PREVIEW=1 to skip the preview window and exit immediately.\n";
}

bool parseThreshold(const char* text, double& out) {
    try {
        std::size_t consumed = 0;
        double value = std::stod(text, &consumed);
        if (consumed != std::string(text).size() || !(value > 0.0) || !(value <= 1.0)) { return false; }
        out = value;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        printUsage(argv[0]);
        return 0;
    }
    if (argc > 5) {
        std::cerr << "Too many arguments.\n\n";
        printUsage(argv[0]);
        return 1;
    }

    // Default input: look for the sample image next to the project root so the
    // program works both when run from build/ and when run from the root.
    std::string readLocation;
    if (argc >= 2) {
        readLocation = argv[1];
    } else {
        for (const char* candidate : {"../images/Sukuna.jpg", "images/Sukuna.jpg"}) {
            if (fs::exists(candidate)) {
                readLocation = candidate;
                break;
            }
        }
        if (readLocation.empty()) {
            std::cerr << "No input image given and no sample image found.\n\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    if (!fs::exists(readLocation)) {
        std::cerr << "Input image not found: " << readLocation << "\n\n";
        printUsage(argv[0]);
        return 1;
    }

    // Default output: alongside the input, with "Canny" appended to the stem.
    std::string writeLocation;
    if (argc >= 3) {
        writeLocation = argv[2];
    } else {
        fs::path out = fs::path(readLocation);
        out.replace_filename(out.stem().string() + "Canny" + out.extension().string());
        writeLocation = out.string();
    }

    double lowerThreshold = 0.03;
    double higherThreshold = 0.1;
    if (argc >= 4 && !parseThreshold(argv[3], lowerThreshold)) {
        std::cerr << "Invalid lower threshold: " << argv[3] << " (expected a value in (0, 1])\n";
        return 1;
    }
    if (argc >= 5 && !parseThreshold(argv[4], higherThreshold)) {
        std::cerr << "Invalid higher threshold: " << argv[4] << " (expected a value in (0, 1])\n";
        return 1;
    }
    if (lowerThreshold > higherThreshold) {
        std::cerr << "lowerThreshold (" << lowerThreshold << ") must not exceed higherThreshold (" << higherThreshold << ").\n";
        return 1;
    }

    const auto start = std::chrono::steady_clock::now();
    const bool ok = cannyEdgeDetection(readLocation, writeLocation, lowerThreshold, higherThreshold);
    if (!ok) {
        return 1;
    }

    // With the preview window open this wall clock would include the time the
    // user spends looking at the result, so only report it when it is headless.
    if (std::getenv("CANNY_NO_PREVIEW") != nullptr) {
        const double totalMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::cout << std::fixed << std::setprecision(2) << "  wall clock (program)   : " << totalMs << " ms\n";
    }

    return 0;
}
