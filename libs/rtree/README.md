# R-Trees: A Dynamic Index Structure for Spatial Searching

## Description

A C++ templated version of [this](http://www.superliminal.com/sources/sources.htm)
RTree algorithm.
The code it now generally compatible with the STL and Boost C++ libraries.
Now, it is compatible with Eigen. See original version of this repo [here](https://github.com/nushoin/RTree)

## Example Usage (Not updated)

```cpp
#include <RTree.h>

// ...

RTree<Foo*, double, 3> tree;
double min[3] = {0., 0., 0.};
double max[3] = {1., 1., 1.};
Foo* bar = new Foo();
tree.Insert(min, max, bar);
```
