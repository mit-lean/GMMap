#!/bin/bash

NN_SIZE=2000
NN_SIZE_LINEAR=500
NN_SIZE_CSCALE=8

die () { echo "FATAL: $1" ; exit 1; }

NINJA="${NINJA:-`which ninja`}" || die "ninja build tool not found"
echo "Ninja found at $NINJA"

CXX="${CXX:-`which c++`}" || CXX=`which clang++` || CXX=`which g++` || die "C++ compile not found"
echo "C++ compiler found at $CXX"

PKG_CONFIG="${PKG_CONFIG:-`which pkg-config`}" || die "pkg-config not found"
echo "pkg-config found at $PKG_CONFIG"

DEPS="eigen3"

VERSIONS=$($PKG_CONFIG --modversion $DEPS) || die "required dependency is missing"

DEPS_ALL=$((printf "%s\n" $DEPS ; $PKG_CONFIG --print-requires $DEPS) | sort | uniq)

CFLAGS="${CFLAGS:--O3 -march=native}"
CFLAGS+=" -std=c++17 -I../src -I../../nigh/src"
CFLAGS+=" $($PKG_CONFIG --cflags-only-I $DEPS_ALL)"
LIBS+=" $($PKG_CONFIG --libs eigen3 $DEPS_ALL)"

if [[ `uname` = Linux ]] ; then
    LIBS+=" -lpthread"
fi

echo "Required dependencies found, generating build.ninja"
exec 3> build.ninja # open build.ninja for writing, use descriptor 3

######################################################################
# Set up the build rules
cat <<-EOF >&3
	ninja_required_version = 1.3
	root = .
	builddir = build
	cxx = $CXX
	cflags = $CFLAGS
	libs = $LIBS

	pool concurrent_pool
	  depth = 1

	rule cxx
	  command = \$cxx -MMD -MT \$out -MF \$out.d \$cflags \$ndebug \$in -o \$out \$libs
	  description = CXX \$out
	  depfile = \$out.d
	  deps = gcc

	rule test
	  command = \$in && touch \$out
	  description = TEST \$in

	rule ctest
	  command = \$in && touch \$out
	  description = CONCURRENCY TEST \$in
	  pool = concurrent_pool

	rule bench
	  command = \$in > \$out
	  description = BENCHMARK \$in
	  pool = concurrent_pool

	rule plot
	  command = ./plot.sh \$out \$title \$in
	  description = PLOT \$out

	EOF

# Generated files are first created as "$1.tmp" then mv'd to "$1" if
# the differ from the existing generated test.  This behavior allows
# the script to be rerun without causing a full rebuild due to
# timestamp changes on the generated files.
replace_tmp () {
    if ! cmp -s "$1" "$1.tmp" ; then
        echo "Generated $1"
        mv "$1.tmp" "$1"
    else
        rm "$1.tmp"
    fi
}

# Adds a test
add_test () {
    tests+=" $test"
    cat <<-EOF >&3
	build \$builddir/$test: cxx $test.cpp
	build \$builddir/$test.success: test \$builddir/$test
	build $test: phony \$builddir/$test.success
	EOF

    cat <<-EOF > "$test.cpp.tmp" 
	// This test case was automatically generated
	// $test.cpp
	$(printf "#include \"../%s.hpp\"\n" $includes)
	#include <nigh/${strat_include}>
	#include "../test_template.hpp"

	TEST($name) {
	    using namespace unc::robotics::nigh;
	    using namespace metric;
	    using namespace nigh_test;
	    using Metric = $metric_type;
	    using State = $store;
	    using Space = metric::Space<State, Metric>;
	    using Concurrency = $concurrency_type;
	    using Strategy = $strategy_type;
	    static constexpr std::size_t N = $N;
	    static constexpr std::size_t K = 50;
	    Space space$args;
	    runTest<$value_type, Space, $member_type, Concurrency, Strategy>(space, N, K);
	}
	EOF

    replace_tmp "$test.cpp"
}

# Adds a concurrency test
add_ctest () {
    name="${space}_${scalar}_${strategy}"
    ctest="$GENDIR/${name}_ctest"
    ctests+=" $ctest"
    cat <<-EOF >&3
	build \$builddir/$ctest: cxx $ctest.cpp
	build \$builddir/$ctest.success: ctest \$builddir/$ctest
	build $ctest: phony \$builddir/$ctest.success
	EOF
    cat <<-EOF > "$ctest.cpp.tmp"
	$(printf "#include \"../%s.hpp\"\n" $includes)
	#include "../concurrent_test_template.hpp"
	#include <nigh/${strat_include}>
	TEST($name) {
	    using namespace unc::robotics::nigh;
	    using Strategy = $strategy_type;
	    using State = $store;
	    using Space = metric::Space<State, ${metric_type}>;
	    Space space$args;
	    nigh_test::runConcurrentTest<Strategy>(space, $N);
	}
	EOF

    replace_tmp "$ctest.cpp"
}

# Adds a benchmark 
add_bench () {
    bench="$GENDIR/${name}_bench"
    benches+=" $bench"
    cat <<-EOF >&3
	build \$builddir/$bench: cxx $bench.cpp
	  ndebug = -DNDEBUG
	build \$builddir/$bench.dat: bench \$builddir/$bench
	build $bench: phony \$builddir/$bench.dat
	EOF

    cat <<-EOF > "$bench.cpp.tmp"
	#include "../bench_template.hpp"
	int main(int argc, char *argv[]) {
	    using namespace unc::robotics::nigh;
	    using namespace unc::robotics::nigh::metric;
	    using namespace nigh_test;
	    using Scalar = $scalar_type;
	    using State = $store;
	    using Metric = $metric_type;
	    using Concurrency = $concurrency_type;
	    using Strategy = $strategy_type;

	    metric::Space<State, Metric> space;
	    runBench<State>(argc, argv, space, nigh_test::Identity{}, Concurrency{}, Strategy{});
	}
	EOF

    replace_tmp "$bench.cpp"
}

######################################################################
# Create test cases for all files that match '*_test.cpp'
tests=""
ctests="" # concurrency tests
for src in *_test.cpp ; do
    base="${src%.*}"
    tests+=" $base"
    cat <<-EOF >&3
	build \$builddir/$base: cxx $src
	build \$builddir/$base.success: test \$builddir/$base
	build $base: phony \$builddir/$base.success
	EOF
done

######################################################################
# Create generated test cases and benchmarks
GENDIR="generated"
mkdir -p $GENDIR
for space in l1_2 l2_3 linf_7 so2_1 so2_7 so3 se3 se3r scaled_se3_7_3 scaled_se3_scalar ; do
    state_list="eigen_vector"
    value_list="state"
    space_args=""
    case $space in
        l*)
            #metric=${space%_*}
            dim=${space#*_}
            metric_type="LP<$dim>"
            includes="sampler_lp"
            [[ $space = "l2_3" ]] && state_list+=" eigen_array std_vector std_array"
            [[ $space = "l2_3" ]] && value_list+=" node nodeptr stateptr"
            ;;
        so2*)
            #metric=${space%_*}
            dim=${space#*_}
            metric_type="SO2<$dim>"
            includes="sampler_so2"
            [[ $space = "so2_1" ]] && state_list+=" scalar";
            ;;
        so3)
            #metric=$space
            dim=4
            metric_type="SO3"
            includes="sampler_so3"
            state_list="eigen_quaternion eigen_vector"
            ;;
        se3*)
            #metric=$space
            dim=-1
            metric_type="Cartesian<SO3, LP<2>>"
            [[ $space = se3r ]] && metric_type="Cartesian<LP<2>, SO3>"
            includes="sampler_lp sampler_so3 sampler_cartesian"
            state_list="tuple pair custom"
            [[ $space = se3r ]] && state_list="tuple"
            ;;
        scaled_se3_7_3)
            #metric=$space
            dim=-1
            metric_type="Cartesian<RatioScaled<SO3, 7>, RatioScaled<LP<2>, 3>>"
            includes="sampler_lp sampler_so3 sampler_cartesian sampler_scaled"
            state_list="tuple"
            ;;
        scaled_se3_scalar)
            dim=-1
            metric_type="Cartesian<Scaled<SO3>, LP<2>>"
            includes="sampler_lp sampler_so3 sampler_cartesian sampler_scaled"
            state_list="tuple"
            space_args="(4.321)"
            ;;
    esac
    
    for state in $state_list ; do
        for scalar in float double long_double ; do
            args="$space_args"
            scalar_type=${scalar//_/ }
            case $state in
                eigen_quaternion) store="Eigen::Quaternion<$scalar_type>" ;;
                eigen_vector)     store="Eigen::Matrix<$scalar_type, $dim, 1>" ;;
                eigen_array)      store="Eigen::Array<$scalar_type, $dim, 1>" ;;
                std_array)        store="std::array<$scalar_type, $dim>" ;;
                std_vector)
                    store="std::vector<$scalar_type>"
                    args="($dim)"
                    ;;
                scalar) store="$scalar_type" ;;
                custom) store="nigh_test::SE3State<$scalar_type>" ;;
                tuple|pair)
                    if [[ $space != se3r ]]
                    then store="std::$state<Eigen::Quaternion<$scalar_type>, Eigen::Matrix<$scalar_type, 3, 1>>" 
                    else store="std::$state<Eigen::Matrix<$scalar_type, 3, 1>, Eigen::Quaternion<$scalar_type>>" ; fi ;;
            esac
            for value in $value_list ; do
                case $value in
                    node)
                        value_type="Node<State>"
                        member_type="NodeKey"
                        ;;
                    state)
                        value_type="State"
                        member_type="Identity"
                        ;;
                    nodeptr)
                        value_type="Node<State>*"
                        member_type="NodePtrKey"
                        ;;
                    stateptr)
                        value_type="State*"
                        member_type="Deref"
                        ;;
                esac
                
                for strategy in batch_8 median linear gnat ; do
                    concurrency_list="rw"
                    [[ $space = l2_3 && $state = eigen_vector && $value = state ]] &&
                        concurrency_list="rw ro nt"
                    case $strategy in
                        linear)
                            strat_include="linear.hpp"
                            strategy_type=Linear
                            ;;
                        batch_*)
                            strat_include="kdtree_batch.hpp"
                            strategy_type="KDTreeBatch<${strategy#*_}>"
                            ;;
                        median)
                            strat_include="kdtree_median.hpp"
                            strategy_type="KDTreeMedian<>"
                            ;;
                        gnat)
                            strat_include="gnat.hpp"
                            strategy_type="GNAT<>"
                            ;;
                    esac
                    
                    for concurrency in $concurrency_list ; do
                        name="${space}_${state}_${scalar}_${value}_${strategy}_${concurrency}"
                        test="$GENDIR/${name}_test"

                        case $concurrency in
                            rw) concurrency_type=Concurrent ;;
                            ro) concurrency_type=ConcurrentRead ;;
                            nt) concurrency_type=NoThreadSafety ;;
                        esac

                        [[ $strategy = linear ]] && N=$NN_SIZE_LINEAR || N=$NN_SIZE

                        add_test

                        if [[ $value = state && $concurrency = rw &&
                              ( ( $space = l2_3  && $state = eigen_vector ) ||
                                ( $space = so2_7 && $state = eigen_vector ) ||
                                ( $space = so3   && $state = eigen_quaternion ) ||
                                ( $space = se3   && $state = tuple ) ||
                                ( $space = scaled_se3_7_3 && $state = tuple) ) ]] ; then

                            if [[ $strategy == linear || $strategy =~ batch_* ]] ; then
                                N=$((N * NN_SIZE_CSCALE))
                            fi

                            add_ctest

                            if [[ $scalar = double ]] ; then
                                add_bench
                            fi                               
                        fi
                    done # for concurrency in ...
                done # for strategy in ...
            done # for value in ...
        done # for scalar in ...
    done # for state in ...
done # for space in ...

######################################################################
# Create plot rules
plots=""
for space in l2_3 so3 so2_7 se3 scaled_se3_7_3 ; do
    plots+=" \$builddir/$space.svg"
    echo -n "build \$builddir/$space.svg: plot " >&3
    # state=eigen_vector
    # case $space in
    #     so3) state=eigen_quaternion ;;
    #     se3) state=tuple ;;
    #     scaled_se3_7_3) state=tuple ;;
    # esac
    for strategy in batch_8 median linear gnat ; do
        echo -n " \$builddir/$GENDIR/${space}_double_${strategy}_bench.dat" >&3
    done
    printf "\n  title = " >&3
    case $space in
        l2_3)  echo "'R^3 with L^2'" >&3 ;;
        so3)   echo "'SO(3)'"        >&3 ;;
        so2_7) echo "'SO(2)x5'"      >&3 ;;
        se3)   echo "'SE(3) 1:1'"    >&3 ;;
        scaled_se3_7_3) echo "'SE(3) 7:3'" >&3 ;;
    esac
done

######################################################################
# Create the main build rules: bench, test, and all
echo "build plots: phony$plots" >&3
echo "build test: phony$tests" >&3
echo "build ctest: phony$ctests" >&3
echo "build all: phony test ctest plots" >&3

exec 3>&- # close build.ninja
echo "Done.  type 'ninja' to start build"
