#############################################################
############ MoM Distance
"""
    dist(data::AbstractVector{T}) where {T<:Union{Tuple{Vararg{<:Real}},Vector{<:Real}}}

Constructs a distance function based on a k-d tree for efficient nearest neighbor search.

# Arguments
- `data::AbstractVector{T}`: A collection of points, where each point is either a tuple of real numbers or a vector of real numbers.

# Returns
- A `DistanceFunction` object that encapsulates the k-d tree and parameters for distance computations.

# Description
This function builds a k-d tree (`KDTree`) from the input data for efficient nearest-neighbor queries. The resulting `DistanceFunction` is structured to use a single nearest neighbor (`k=1`) and is labeled with `type="dist"`.

# Example Usage
```julia
data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
distance_func = dist(data)
```
"""
function dist(
    data::AbstractVector{T}
) where {T<:Union{Tuple{Vararg{<:Real}},Vector{<:Real}}}

    Xq = [data]

    trees = [KDTree(reduce(hcat, xq), leafsize=1) for xq in Xq]

    return DistanceFunction(
        k=1,
        trees=trees,
        X=Xq,
        type="dist",
        Q=1
    )
end



#############################################################
############ DTM

"""
    dtm(data::AbstractVector{T}, m::Real) where {T<:Union{Tuple{Vararg{<:Real}},Vector{<:Real}}}

Constructs a distance-to-measure (DTM) function using a k-d tree for nearest neighbor computations.

# Arguments
- `data::AbstractVector{T}`: A collection of points, where each point is either a tuple of real numbers or a vector of real numbers.
- `m::Real`: A proportion parameter (between 0 and 1) used to determine the number of neighbors for the DTM computation.

# Returns
- A `DistanceFunction` object representing the DTM, built using a k-d tree for efficient nearest-neighbor searches.

# Description
The function builds a `KDTree` from the given data and computes a distance-to-measure function. The number of neighbors used for the computation is determined by `m * length(data)`, rounded down to the nearest integer. The function returns a `DistanceFunction` with `type="dtm"` for further analysis.

# Example Usage
```julia
data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
m = 0.2
dtm_func = dtm(data, m)
```
"""
function dtm(
    data::AbstractVector{T},
    m::Real
) where {T<:Union{Tuple{Vararg{<:Real}},Vector{<:Real}}}

    tree = KDTree(reduce(hcat, data), leafsize=1)

    return DistanceFunction(
        k=floor(Int, m * length(data)),
        trees=[tree],
        X=[data],
        type="dtm",
        Q=1
    )
end


#############################################################
############ MoM DTM


function momdtm(
    data::AbstractVector{T},
    m::Real,
    Q::Integer=0
) where {T<:Union{Tuple{Vararg{<:Real}},Vector{<:Real}}}

    if Q < 1
        println("Invalid value of Q supplied. Defaulting to Q = n_obs / 5 = $Q")
        Q = ceil(Int16, length(data) / 5)
    end

    Xq = [fold[2] for fold in kfolds(shuffleobs(data), Q)]

    trees = [KDTree(reduce(hcat, xq), leafsize=1) for xq in Xq]

    return DistanceFunction(
        k=floor(Int, m * length(data)),
        trees=trees,
        X=Xq,
        type="momdtm",
        Q=Q
    )
end


#############################################################
############ MoM Distance
"""
    momdist(data::AbstractVector{T}, Q=0) where {T<:Union{Tuple{Vararg{<:Real}},Vector{<:Real}}}

Constructs a median-of-means distance function using k-d trees for robust nearest neighbor computations.

# Arguments
- `data::AbstractVector{T}`: A collection of points, where each point is either a tuple of real numbers or a vector of real numbers.
- `Q::Int` (optional): The number of partitions (folds) for the median-of-means method. If `Q < 1`, it defaults to `length(data) / 5`.

# Returns
- A `DistanceFunction` object representing the median-of-means distance function, utilizing k-d trees for efficient nearest-neighbor searches.

# Description
This function partitions the dataset into `Q` folds, shuffles the data, and constructs a `KDTree` for each fold. The k-d trees facilitate efficient nearest-neighbor queries for robust distance estimation. If `Q` is not provided or invalid, it defaults to `n_obs / 5`.

# Example Usage
```julia
data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
momdist_func = momdist(data, Q=2)
```
"""
function momdist(
    data::AbstractVector{T},
    Q=0
) where {T<:Union{Tuple{Vararg{<:Real}},Vector{<:Real}}}

    if Q < 1
        println("Invalid value of Q supplied. Defaulting to Q = n_obs / 5 = $Q")
        Q = ceil(Int16, length(data) / 5)
    end

    Xq = ThreadsX.collect(fold[2] for fold in kfolds(shuffleobs(data), Q))

    trees = ThreadsX.collect(KDTree(reduce(hcat, xq), leafsize=1) for xq in Xq)

    return DistanceFunction(
        k=1,
        trees=trees,
        X=Xq,
        type="momdist",
        Q=Q
    )
end

"""
    parallel_momdist(X)

Computes the Median of Means (MoM) distance metric in parallel using multiple worker processes.

# Arguments
- `X::AbstractMatrix`: The dataset for which the Median of Means (MoM) distance is to be computed.

# Returns
- `DistanceFunction`: A custom distance function object containing:
  - `k`: The number of nearest neighbors (fixed at 1).
  - `trees`: A collection of trees built on different folds of `X` for parallel computation.
  - `X`: The original dataset wrapped in an array.
  - `type`: A string indicating the type of distance function (`"momdist"`).
  - `Q`: The number of worker processes used for parallel computation.

# Description
This function partitions the dataset `X` into `m` folds (where `m` is the number of available worker 
processes), builds a tree structure for each partition in parallel using `pmap`, and returns a 
distance function object. The Median of Means (MoM) method enhances robustness by splitting the dataset 
into multiple groups, computing the mean within each, and using the median of these means to reduce the 
impact of outliers.

# Example
```julia
addprocs(4)  # Add worker processes for parallel computation
@everywhere using RobustTDA
X = rand(100, 2)  # Generate a dataset with 100 points in 2D space
momdist_func = parallel_momdist(X)  # Compute parallel Median of Means distance metric
```
"""
function parallel_momdist(X)
    m = nworkers()
    Xq = collect(reduce(hcat, fold[2]) for fold in kfolds(shuffleobs(X), m))
    trees = pmap(make_tree, Xq)
    return DistanceFunction(
        k=1,
        trees=trees,
        X=[X],
        type="momdist",
        Q=m
    )
end

function make_tree(x)
    return KDTree(x, leafsize=1)
end

function tree_dist(tree, X)
    return knn(tree, X, 1)[2] |> Base.Flatten |> collect
end


#############################################################
############ Fit for Distance Functions

"""
    fit(x::AbstractVecOrMat, D::DistanceFunction)

Computes the median of k-nearest neighbor distances for a given dataset using a precomputed distance function.

# Arguments
- `x::AbstractVecOrMat`: The input data points for which distances are computed.
- `D::DistanceFunction`: A precomputed distance function containing k-d trees and related parameters.

# Returns
- A vector of median distances for each input data point.

# Description
This function computes k-nearest neighbor distances for each data point in `x` using the k-d trees stored in `D`. It averages the distances across different k-d trees and returns the median distance for each data point. If the number of available neighbors is smaller than `k`, it adjusts `k` accordingly to avoid errors.

# Example Usage
```julia
x = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
D = dist(x)  # Assume `dist` function constructs a DistanceFunction
fit_values = fit(x, D)
println(fit_values)
```
"""
function fit(
    x::AbstractVecOrMat,
    D::DistanceFunction
)
    @unpack k, trees, X, type, Q = D

    if length(X[1]) < k
        k = ([length(y) for y in X] |> minimum)
    end

    xq = hcat(x...)
    dists = ThreadsX.collect(knn(tree, xq, k)[2] .|> mean for tree in trees)
    dists = reduce(hcat, dists)
    return median.(eachrow(dists))
    # fit = []
    # for j ∈ eachindex(x)
    #     push!(fit,
    #         ThreadsX.collect(maximum(knn(trees[i], x[j], k)[2]) for i = 1:Q)
    #     )
    # end
    # return reduce(vcat, median.(fit))
end

"""
    parallel_fit(X, df)

Computes the median of nearest neighbor distances using multiple KDTree structures in parallel.

# Arguments
- `X`: A vector of data points, where each point is a vector.
- `df`: A `DistanceFunction` object containing multiple KDTree structures.

# Returns
- A column vector where each entry represents the median nearest neighbor distance for a data point.

# Example
```julia
X = [rand(2) for _ in 1:100]  # 100 points in 2D
df = parallel_momdist(X)  # Create distance function using KDTree
distances = parallel_fit(X, df)  # Compute median nearest neighbor distances
```
"""
function parallel_fit(X, df)
    X_mat = reduce(hcat, X)
    dists = hcat(pmap(tree -> tree_dist(tree, X_mat), df.trees)...)
    return median(dists, dims=2)
end