<div align="center">
  <img src="docs/src/assets/logo.svg" alt="RobustTDA.jl" width="480">

_Robust Topological Data Analysis in Julia._

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sidv23.github.io/RobustTDA.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sidv23.github.io/RobustTDA.jl/dev)
[![Build Status](https://github.com/sidv23/RobustTDA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sidv23/RobustTDA.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/sidv23/RobustTDA.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/sidv23/RobustTDA.jl)

</div>


# RobustTDA.jl

`RobustTDA.jl` provides a robust and flexible framework for computing persistent homology from point-cloud data in the presence of noise. RobustTDA uses the blazing fast ![Ripserer.jl](https://github.com/mtsch/Ripserer.jl) backend for computing persistent homology. 

Please see the ![documentation]() for further information, usage and examples.

# Getting Started

### Installation
From the Julia REPL, type `]` to enter the Pkg REPL mode and run
```julia
    pkg> add https://github.com/sidv23/RobustTDA.jl
```

### Example

Generate some data

```julia
using RobustTDA
using Pipe

# Signal from a circle with noise
signal = 2 .* randCircle(500, sigma = 0.05)

# Outliers from a Matérn cluster process
win = (-1, 1, -1, 1)
noise = randMClust(100, window = win, λ1 = 2, λ2 = 10, r = 0.05)

X = [signal; noise]
points = [signal; noise]
scatter(points)
```

![Points](docs/src/assets/points.svg)

First, convert the data to a vector-format, i.e.,
```julia
Xn = [[x...] for x in points]
```
<sub><sup>*All rountines in the package currently read the input point-cloud data as a vector of vectors</sub></sup>


```julia
plt = f -> plot(xseq, yseq, (x,y) -> RobustTDA.fit([[x,y]],f), st=:surface)
```

Construct a filter function from the samples $\mathbb{X}_n$. For example,

1. Vanilla Distance function:
    ```julia
        f_dist = dist(Xn)
        plt(f_dist)
        
        # RobustTDA.DistanceFunction
        # k: Int64 1
        # trees: Array{KDTree{StaticArrays.SVector{2, Float64}, Euclidean, Float64}}((1,))
        # X: Array{Vector{Vector{Float64}}}((1,))
        # type: String "dist"
        # Q: Int64 1
    ```
    <img src="./docs/src/assets/dist.svg" width=50% align="center">
2. Distance-to-measure:
    ```julia
        f_dtm = dtm(Xn, 0.1)
        plt(f_dtm)

        # RobustTDA.DistanceFunction
        # k: Int64 60
        # trees: Array{BruteTree{StaticArrays.SVector{2, Float64}, Euclidean}}((1,))
        # X: Array{Vector{Vector{Float64}}}((1,))
        # type: String "dtm"
        # Q: Int64 1
    ```
    <img src="./docs/src/assets/dtm.svg" width=50% align="center">
2. Median-of-means Distance Function:
   ```julia
        f_momdist = momdist(Xn, 101)
        plt(f_momdist)

        # RobustTDA.DistanceFunction
        # k: Int64 1
        # trees: Array{KDTree{StaticArrays.SVector{2, Float64}, Euclidean, Float64}}((201,))
        # X: Array{SubArray{Vector{Float64}, 1, Vector{Vector{Float64}}, Tuple{Vector{Int64}}, false}}((201,))
        # type: String "momdist"
        # Q: Int64 201
    ```
     <img src="./docs/src/assets/momdist.svg" width=50% align="center">



For sublevel persistent homology, you need to specify a grid to evaluate the filter functions
```julia
    xseq = -3:0.05:3
    yseq = -3:0.05:3
    F_grid = [RobustTDA.fit([[x, y]], f) for x in xseq, y in yseq]
    D = F_grid |> Cubical |> ripserer
```
