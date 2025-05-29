This directory contains files demonstrating how to use Nigh.  CMake is required to build the demo files.  Eigen 3 and a C++ 17 capable compiler is required.

To build:

    % mkdir build
    % cmake ..
    % make

If the default compiler does not support C++ 17, use the following commands (assuming, for example, that the C++ 17 compiler is in the path and named 'clang++-6.0'.  Substitute your compiler as appropriate):

    % mkdir build
    % CXX=clang++-6.0 cmake ..
    % make

The following demo files are included:

euclidean_demo.cpp
=============

This demo generates random 2D points on a 1.0x1.0 plane, performs a single query, then writes the output to an image file.  You can modify the run-time behavior of the command with the following options:

-o OUT -- generates OUT (default euclidean_demo.svg)
-k COUNT -- compute the k nearest neighbors (default is 1, unless -r is specified)
-r RADIUS -- compute the nearest neighbors in radius
-N COUNT -- generate COUNT random points (default is 100)

After running this demo, open 'euclidean_demo.svg' in any modern browser to see the random points and the result of the query.  Note: only minimal argument checks on the arguments are performed.


custom\_vector\_demo.cpp
================

This demo shows how to use a custom vector type as the key type in a nearest neighbor structure.  Out of the box Nigh supports Eigen3 vectors, vectors based upon std::vector and std::array.  However if a project is using a custom vector type or a vector based upon a different linear algebra library, Nigh can support it.  The only requirement is a template specialization.

When run, this demo will output the 10 nearest neighbors of 100 random points in a Euclidean R^3 space using a custom vector type.
