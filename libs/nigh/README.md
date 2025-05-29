# nigh
Nigh is a *concurrent* exact nearest neighbor searching library for Euclidean, SO(3), SE(3), SO(2), SE(2), and weighted combinations thereof.  As a concurrent data structure it supports multiple threads concurrently inserting and querying the data structure with minimal wait time.  As such it is ideal for embedding into parallel algorithms which require sharing a nearest neighbor data structure between multiple threads.

Nigh is a header-only library.  Its only dependency (Eigen) is also a header-only library.  Installation thus requires copying the header files into a known location.

### Requirements 
   * C++ 17 compiler (tested in clang 5+, gcc 7+)
   * Eigen
   * gnumake for tests
   
## Usage

### Defining a `MetricSpace`

Every Nigh object has a *key* type that forms a basis for the metric space of the nearest neighbor searching.  The key type stores the state from the space.  For this example, we will walk through searching for nearest neighbors in a Euclidean metric on R^3.  We start with the definition of the state type.  In this case we will use an 3-dimensional vector from Eigen (though other types are available out of the box).  We will define a type for the state:

```c++
using State = Eigen::Vector3d;
```

Next we define the metric that we wish to use.  In this case we wish to use a Euclidean metric, or L^2 metric, and thus we can use:

```c++
using Metric = nigh::LPMetric<2>;
```

Finally we put it together into a MetricSpace:

```c++
using Space = MetricSpace<State, Metric>;
```

### Stored values and the key function

Nigh supports storing objects of arbitrary types, and requires a suitable function that can return the key from the object.  For example, if an application stores and searches for objects of a custom node type:

```c++
struct MyNode {
    State key_;
    std::string name_;
  
    // using default copy constructor
    MyNode(const State& key, const std::string& name)
        : key_(key), name_(name)
    {
    }
};
```

We can define the key function for this object using a functor:

```c++
struct MyNodeKey {
    const State& operator() (const MyNode& node) const {
        return node.key_;
    }
};
```

### Nearest Neighbor Template

With the stored type, metric space, and key function defined, we can create a nearest neighbor searching structure:

```c++
Nigh<MyNode, Space, MyNodeKey> nn;
```
# Supported Metrics

Metrics are represented by tagging types and templates.

Type/Template | Metric
------------- | ------
`LPMetric<p>`   | L^p metric, equivalent to `pow(pow(a[i] - b[i], p) + ..., 1/p)`.  Values in this space are represented as a collection of scalars, e.g. Eigen::Matrix, Eigen::Array, std::array, or std::vector.
`L2Metric`      | Alias for `LPMetric<2>`, the Euclidean metric
`L1Metric`      | Alias for `LPMetric<1>`, the Manhattan metric
`LInfMetric`    | Alias for `LPMetric<-1>`, L^Infinity metric
`SO2Metric<p>`  | SO(2) rotational distance in a plane.  When the space is R^n (i.e., multiple rotational values), the distance is summed as it would be with same `p` in `LPMetric<p>`.
`SO3Metric`     | SO(3) rotational distance defined as the shorter of two angle along the great arc that subtends the two values.  Currently this assumes that the space will be represented by a unit quaternion.
`CartesianMetric<M...>` | A metric that is the sum of contained metrics.
`ScaledMetric<M, std::ratio<num, den>>` | A metric (`M`) scaled by a particular ratio.  This is meant to be used within a Cartesian metric.
`ScaledMetric<M>` | A metric (`M`) scaled by a scalar value at runtime.  This metric's constructor requires a scalar value argument.

## Using Scaled Metrics

The `ScaledMetric` templates wrap another metric so that the distance function is multiplied by a scalar value.  As multiplying the distance by a scalar does not change the relation between pairs of points, the `ScaledMetric` is primarily for use embedded within a `CartesianMetric`.  Currently Nigh supports two forms of scaled metrics: (1) scaled by a compile-time constant as determined by a `std::ratio`, and (2) scaled by a runtime scalar stored in an instance of a space.  The former should be preferred when the constant is fixed as it does not require additional storage or maintainence of the scalar at runtime.  The later should be used when the scalar value can be configured at runtime.  But note that it is not possible to change the scalar once a the space is used by the tree as the scalar will be copied into a private variable at construction time.

Example ratio weighted space:

```c++
using namespace unc::robotics::nigh;
using Weight = std::ratio<7, 2>;
using Space = CartesianSpace<
    ScaledSpace<SO3Space<double>, std::ratio<7>>,    // multiply SO(3) distance by 7
    ScaledSpace<L2Space<double>, std::ratio<2,3>>>;  // multiply L2 distance by 0.6666
Space space;
```

Note, if you have a decimal ratio at compile time, just set the `std::ratio` denominator to the appropriate power of 10.  For example, to use a ratio of 12.345, use `std::ratio<12345,1000>`.

Example runtime scalar weighted space:

```c++
using namespace unc::robotics::nigh;
using Space = CartesianSpace<
    ScaledSpace<SO3Space<double>>,
    ScaledSpace<L2Space<double, 3>>>;
// weighting scalar type matches scalar type of underlying space
double so3wt = 12.345;     
double l2wt = 6.789;
Space space(so3wt, l2wt);
Nigh<..., Space, ...> nn(space);
```

See also examples in the test directory.

# Predefined Spaces

Nigh includes some pre-defined `MetricSpace` templates for common scenarios.

Type/Template | Metric | Type
------------- | ------ | ----
`LPSpace<S, dim, p>` | `LPMetric<p>` | `Eigen::Matrix<S, dim, 1>`
`SE2Space<S>` | `CartesianMetric<LPMetric<2>, SO2Metric<>>` | `std::tuple<Eigen::Matrix<S, 2, 1>, S>`
`SE3Space<S>` | `CartesianMetric<SO3Metric, LPMetric<2>>` | `std::tuple<Eigen::Quaternion<S>, Eigen::Matrix<S, 3, 1>>`

Note: SE(2) and SE(3) types are organized for better type alignment.

# Nearest Neighbor API

```c++
template <
    T,
    Space,
    KeyFn,
    Concurrency,
    Strategy,
    Allocator>
class Nigh;
```

## Member Types

Member Type | Definition
----------- | ----------
`Type` | The `T` parameter 
`Space` | The `Space` parameter
`Key` | The key type from the metric space
`Metric` | The metric type from the metric space
`Distance` | The distance type from the metric space

## Member Functions

### Constructors
```c++
Nigh(const Space& space = Space(), const KeyFn& keyFn = KeyFn(), const Allocator& alloc = Allocator())
```
This is the default and main constructor.  It stores a copy of the arguments, and initializes the nearest neighbor structure to an empty structure.

```c++
Nigh(Nigh&&)
```
The move constructor.  All nearest-neighbor structures are movable of their types are movable.   Typically moves are fast O(1) operations that leave the source of the move in a valid, but empty state.

```c++
Nigh(const Nigh&) = delete;
```
*Currently there are copy constructors.*  This may change later, but it is an expensive operation that may be better served by algorithms that rebalance the resulting copy in the process.

### Mutation

```c++
void insert(const Type& value);
```
Inserts a value into the nearest neighbor data structure.  The argument will be copied into the data structure.  This operation is thread-safe only if the `Concurrency` template argument is `Concurrent`.  Otherwise mutual exclusion from all other concurrent operation is must be insured by the caller (e.g., by locks or by only using a single thread).

```c++
void clear();
```
Removes all values from the nearest neighbor structure, setting the size to 0.  This is *never* a thread-safe operation, regardless of `Concurrency` setting.

### Searching

```c++
std::optional<std::pair<Type, Distance>> nearest(const Key& key) const;
```
Returns the nearest value to, and its distance from, the argument.  The return value is a `std::optional` that will always be present unless the data structure is empty.

```c++
std::optional<Type> nearest(const Key& key, Distance *distOut) const;
```
Returns the nearest value to, and its distance from, the argument.  The return value is a `std::optional` that will always be present unless the data structure is empty.  When a the result is present, the value in `distOut` will be the distance between the key and the result.  The `distOut` parameter may be `nullptr`, in which case the distance will is not returned.

```c++
template <typename Tuple, typename ResultAllocator>
void nearest(
    std::vector<Tuple, ResultAllocator>& nbh,
    const Key& key,
    std::size_t k,
    Distance maxRadius = std::numeric_limits<Distance>::infinity()) const;
```
Searches for the `k`-nearest neighbors of key, optionally within a constrained radius.  The nearest neighbors are returned in the `nbh` vector provided.  The `Tuple` type may be any tuple-like object with a `Type` and a `Distance` member in any order.  Thus `std::pair<Type, Distance>` and `std::tuple<Distance, Type>` are valid types for `Tuple`.  In this case, "tuple-like" means that `std::get<I>` and `std::tuple_element<I, Tuple>` are defined.

This method first clears the `nbh` argument, then populates it with the nearest neighbors.  The caller thus does *not* need to call `nbh.clear()` between calls to `nearest`.  It may however be advisable to call `nbh.reserve(k+1)` before calling this method to avoid unneccessary allocations within `std::vector` during the nearest-neighbor search.  The `+1` part is required since `nearest()` will temporarily increase the size to `k+1` during searching.  The final result will contain at most `k` values.

This method also supports searching for all neighbors within a radius.  To get this behavior, set the `k` parameter to `nbh.max_size()`.

```c++
std::vector<std::pair<Type, Distance>> nearest(
    const Key& key,
    std::size_t k,
    Distance maxRadius = std::numeric_limits<Distance>::infinity()) const;
```
This method is convience wrapper for the other `k`-nearest neighbor search method.  It performs the same searching, and has the same result meaning for `key`', `k`, and `maxRadius` parameters.

### Other

```c++
std::size_t size() const;
```
Returns the current number of values in the data structure.

```c++
allocator_type get_allocator() const;
```
Returns the allocator.  This method signature matches STL's.

```c++
const Space& metricSpace() const;
```
Returns the metric space.

