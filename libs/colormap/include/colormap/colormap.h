//
// Created by peter on 4/3/22.
// Adapted from https://github.com/tdegeus/cppcolormap to use the eigen interface only!
// Stores the color basesets

#ifndef COLORMAP_H
#define COLORMAP_H
#include "colormap/seqcolorbase.h"
#include "colormap/divcolorbase.h"
#include "colormap/pltcolorbase.h"
#include <string>
#include <stdexcept>

namespace colormap {
    inline Eigen::Vector3d interpolateNearestNeighbor(const Eigen::MatrixXd& scaledColorMap, double percentage){
        // Performs nearest neighbor interpolation
        if (percentage > 1 || percentage < 0 || std::isnan(percentage)){
            throw std::invalid_argument( fmt::format("Invalid percentage: {:.2f}", percentage));
        }
        int idx = (int) round((scaledColorMap.rows() - 1) * percentage);
        return scaledColorMap.row(idx);
    }

    inline Eigen::Vector3d interpolate(const Eigen::MatrixXd& scaledColorMap, double percentage){
        // Perform linear interpolation
        if (percentage > 1 || percentage < 0){
            throw std::invalid_argument( "Invalid percentage range!");
        }

        double frac_idx = (scaledColorMap.rows() - 1) * percentage ;
        int int_floor = (int) floor(frac_idx);
        int int_ceil = (int) ceil(frac_idx);
        if (int_floor == int_ceil){
            return scaledColorMap.row(int_floor);
        } else {
            auto delta = scaledColorMap.row(int_ceil) - scaledColorMap.row(int_floor);
            return scaledColorMap.row(int_floor) + (frac_idx - int_floor) * delta;
        }
    }

    inline Eigen::MatrixXd resizeColorMap(const Eigen::MatrixXd& baseColorMap, int N){
        // Construct a colormap
        if (baseColorMap.rows() == N || N <= 0){
            return baseColorMap;
        }

        Eigen::MatrixXd intColorMap(N, 3);
        for (int i = 0; i < N; i++){
            intColorMap.row(i) = interpolate(baseColorMap, (double) i / (N-1.0));
        }
        return intColorMap;
    }

    inline Eigen::MatrixXd createColorMap(const std::string& colorClass, const std::string& colorName, int N){
        // Construct a colormap
        if (colorClass == "sequential"){
            Eigen::MatrixXd baseColorMap = sequential::colorbase.at(colorName).cast<double>() / 255.0;
            return resizeColorMap(baseColorMap, N);
        } else if (colorClass == "diverging") {
            Eigen::MatrixXd baseColorMap =  div::colorbase.at(colorName).cast<double>() / 255.0;
            return resizeColorMap(baseColorMap, N);
        } else if (colorClass == "matplotlib") {
            return plt::colorbase.at(colorName).constructMap(N);
        } else {
            throw std::invalid_argument( "Invalid colormap class!");
        }
    }
}

#endif //COLORMAP_H
