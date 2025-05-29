# Unit Test for MPT

Running the unit tests requires [ninja](https://ninja-build.org/).  Nigh requires [Eigen 3](http://eigen.tuxfamily.org).

    % ./configure.sh
    % ninja test

Generating the benchmark plots requires gnuplot.

    % ninja plot

Test classes are in `*_test.cpp` files.  The `./configure.sh` script finds files that match this pattern and generates the build rules to run the tests.  Whenever a new test is added the script will have to be run again.

The `./configure.sh` script generates tests to exercise template variants.  These tests will appear in the `generated` folder after the script is run.

Some `./configure.sh` script behaviors can be overridden with environment variables.  To set the C++ compiler, use the `CXX` environment variable.  To set the flags the compile will use, set the `CFLAGS` environment variable.   Here are a few examples:

To generate a build file that will use clang++ and compile in debug symbols:

    % CXX=clang++ CFLAGS=-g ./configure.sh

To generate a build file that will use g++ using the default compiler flags:

    % CXX=g++ ./configure.sh

Some compilers require an additional library to support atomic operations on long double.  For those, add `-latomic` to the list of libraries.  This situtation will be made apparent by the compiler link messages similar to "`undefined reference to __atomic_store_16`"

    % CXX=g++ LIBS=-latomic ./configure.sh

