#############################################################
############ Balls

"""
    bandwidth_select(Xn, k)

Estimates a bandwidth parameter for density estimation or clustering using k-nearest neighbors.

# Arguments
- `Xn::Vector{Vector{<:Real}}`: A vector of data points, where each element is a coordinate vector.
- `k::Int`: The number of nearest neighbors to consider. If `k < 2`, it is set to `2` to ensure meaningful results.

# Returns
- `Float64`: The median of the maximum distances to the `k`-th nearest neighbor for each data point in `Xn`.

# Description
This function constructs a `KDTree` for efficient nearest neighbor search and computes the `k`-th nearest neighbor distance for each data point. The final bandwidth estimate is obtained by taking the median of these distances, providing a robust measure of local density.

# Example
```julia
# Generate random 2D points
Xn = [rand(2) for _ in 1:100]

# Select bandwidth using 5-nearest neighbors
bw = bandwidth_select(Xn, 5)
```
"""
function bandwidth_select(Xn, k)

    if k < 2
        k = 2
    end

    kdtree = KDTree(reduce(hcat, Xn), leafsize=1)
    knns = [knn(kdtree, Xn[i], k)[2] |> maximum for i = eachindex(Xn)]
    return median(knns)
end


#############################################################
############ Balls
"""
    Balls(X, R)

Creates a collection of `Ball2` objects centered at points in `X` with corresponding radii from `R`.

# Arguments
- `X::Vector{Vector{<:Real}}`: A vector of coordinate points, where each element is a coordinate vector.
- `R::Union{Real, Vector{<:Real}}`: The radius or a vector of radii for the balls.
  - If `R` is a single value, all balls will have the same radius.
  - If `R` is a vector, each ball gets a corresponding radius from `R`.

# Returns
- `Vector{Ball2}`: A list of `Ball2` objects, each defined by a center from `X` and a radius from `R`.

# Description
This function generates `Ball2` objects, which represent balls (discs in 2D, spheres in 3D, etc.), given a set of center points `X` and radii `R`. If a single radius is provided, all balls will have the same size; otherwise, each ball gets its respective radius from `R`.

# Example
```julia
# Define centers
X = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]

# Define a single radius
balls1 = Balls(X, 0.5)

# Define individual radii
radii = [0.5, 1.0, 1.5]
balls2 = Balls(X, radii)
```
"""
function Balls(X, R)
    if length(R) == 1
        return [Ball2([x...], float(R)) for x in X]
    else
        return [Ball2([x...], float(r)) for (x, r) in zip(X, R)]
    end
end


#############################################################
############ Radius Function
"""
    rfx(t, w, p)

Computes a transformation based on the values of `t`, `w`, and the parameter `p`.

# Arguments
- `t::Real`: Input value or array of values.
- `w::Real`: Reference value or array of values.
- `p::Real`: Power parameter (can be `Inf` for a special case).

# Returns
- If `t > w`:
  - When `p != Inf`: Computes `(t^p - w^p)^(1/p)`.
  - When `p == Inf`: Returns `t`.
- Otherwise, returns `0`.

# Description
This function applies a conditional transformation on `t` based on `w` and `p`. It is commonly used in geometric or statistical computations where distance-like measures are required.

# Example
```julia
rfx(3, 2, 2)    # Returns sqrt(3^2 - 2^2) = sqrt(9 - 4) = sqrt(5)
rfx(2, 3, 2)    # Returns 0, since t is not greater than w
rfx(4, 2, Inf)  # Returns 4, as p == Inf
```
"""
function rfx(t, w, p)
    return t .> w ? (p != Inf ? (t^p .- w .^ p) .^ (1 / p) : t) : 0
end



#############################################################
############ Convert Distance
"""
    convert_radius(rad, k)

Converts a given radius `rad` into an equivalent transformed radius based on the kernel `k`.

# Arguments
- `rad::Real`: The input radius value to be converted.
- `k::Kernel`: The kernel function used for the transformation.

# Returns
- A transformed radius value, computed based on the specific kernel type.

# Description
This function maps a given radius `rad` to a corresponding transformed value using different kernel types. The transformation varies depending on the kernel:
  - `SqExponentialKernel`: Uses `sqrt(-2 * log(t0)) * scale`
  - `ExponentialKernel`: Uses `-log(t0) * scale`
  - `GammaExponentialKernel`: Uses `(-log(t0))^(1 / γ) * scale`
  - `RationalKernel`: Uses `α * ((t0^(-1/α)) - 1) * scale`
  - `RationalQuadraticKernel`: Uses `sqrt(2 * α * ((t0^(-1/α)) - 1)) * scale`
  - `GammaRationalKernel`: Uses `(α * ((t0^(-1/α)) - 1))^(1/γ) * scale`

If `t0 <= 0`, the function returns `√2` as a default value. If the kernel type is unsupported, a `DomainError` is thrown.

# Example
```julia
k = SqExponentialKernel()
convert_radius(1.5, k)  # Returns the transformed radius
```
"""
function convert_radius(rad, k)

    # r0 = Roots.find_zero(t -> dH(k)(0, t) - rad, 0., atol=eps(1.))
    # r0 = Roots.find_zero(t -> dH(k)(0, t) - rad, 0., rtol=1e-10)
    t0 = (2 - ((rad)^2)) / 2
    scale = 1 / (k.transform.s...)

    if t0 <= 0
        return √2
    else
        if typeof(k.kernel.kernel) <: SqExponentialKernel
            return sqrt(-2 * log(t0)) * scale

        elseif typeof(k.kernel.kernel) <: ExponentialKernel
            return -log(t0) * scale

        elseif typeof(k.kernel.kernel) <: GammaExponentialKernel
            return (-log(t0))^(1 / k.kernel.kernel.:γ...) * scale

        elseif typeof(k.kernel.kernel) <: RationalKernel
            return ((k.kernel.kernel.:α...) * ((t0^(-1 / k.kernel.kernel.:α...)) - 1)) * scale

        elseif typeof(k.kernel.kernel) <: RationalQuadraticKernel
            return sqrt((2 * k.kernel.kernel.:α...) * ((t0^(-1 / k.kernel.kernel.:α...)) - 1)) * scale

        elseif typeof(k.kernel.kernel) <: GammaRationalKernel
            return ((k.kernel.kernel.:α...) * ((t0^(-1 / k.kernel.kernel.:α...)) - 1))^(1 / k.kernel.kernel.:γ...) * scale

        else
            throw(DomainError(k, "Kernel Type Unsupported"))
        end
    end
end



#############################################################
############ Filtration value of Vertices and Edges

# function weighted_filtration_value(d, wx, wy, p)
#     if d < (wx^p - wy^p)^(1 / p)
#         return maximum([wx, wy])
#     else
#         if p == 1
#             return 0.5 * (wx + wy + d)
#         elseif p == 2
#             return √(((wx + wy)^2 + d^2) * (abs(wx - wy)^2 + d^2)) / (2 * d)
#         elseif p == Inf
#             return maximum([wx, wy, d / 2])
#         end
#     end
# end
"""
    weighted_filtration_value(d, wx, wy, p)

Computes the weighted filtration value based on distance `d` and weights `wx`, `wy`, using the parameter `p`.

# Arguments
- `d::Real`: The distance between two points.
- `wx::Real`: The weight associated with the first point.
- `wy::Real`: The weight associated with the second point.
- `p::Real`: The norm parameter (can be `1`, `2`, or `Inf`).

# Returns
- A computed filtration value based on the chosen norm `p`.

# Description
This function calculates the weighted filtration value using different formulas depending on `p`:
- If `p == 1`:
  - If `d ≤ |wx - wy|`, returns `max(wx, wy)`.
  - Otherwise, returns `0.5 * (wx + wy + d)`.
- If `p == 2`:
  - If `d ≤ √|wx² - wy²|`, returns `max(wx, wy)`.
  - Otherwise, returns:
    ```julia
    sqrt(((wx + wy)^2 + d^2) * (|wx - wy|^2 + d^2)) / (2 * d)
    ```
- If `p == Inf`, returns `max(wx, wy, d)`.
- If `p` is not `1`, `2`, or `Inf`, the function throws an error.

# Example
```julia
weighted_filtration_value(1.5, 0.8, 1.2, 2)  # Computes the value using p = 2
```
"""
function weighted_filtration_value(d, wx, wy, p)
    if p == 1
        if d ≤ abs(wx - wy)
            return maximum([wx, wy])
        else
            return 0.5 * (wx + wy + d)
        end
    elseif p == 2
        if d ≤ √(abs(wx^2 - wy^2))
            return maximum([wx, wy])
        else
            return √(((wx + wy)^2 + d^2) * (abs(wx - wy)^2 + d^2)) / (2 * d)
        end
    elseif p == Inf
        return maximum([wx, wy, d])
    else
        error("Invalid value of p")
    end
end



#############################################################
############ Weighted Rips Filtration
"""
    wrips(Xn; w=nothing, ρ=Euclidean(1e-12), p=1, type="points", args...)

Computes the weighted Vietoris–Rips filtration for a given dataset.

# Arguments
- `Xn::Matrix{<:Real} | Matrix{<:Real}`: The input dataset, where rows represent points.
- `w::Vector{<:Real}=nothing`: Optional weight vector; if not provided, defaults to zeros.
- `ρ::Metric=Euclidean(1e-12)`: Distance metric used for computing pairwise distances.
- `p::Real=1`: Parameter controlling the filtration computation (e.g., `1`, `2`, or `Inf`).
- `type::String="points"`: Determines input format:
  - `"points"` (default) → Computes pairwise distances.
  - `"distMatrix"` → Uses `Xn` directly as a distance matrix.
- `args...`: Additional arguments passed to `ripserer`.

# Returns
- `Symmetric` distance matrix modified with weighted filtration values, passed to `ripserer`.

# Description
1. If `type == "points"`, computes the pairwise distance matrix `D` using metric `ρ`.
2. If `type == "distMatrix"`, assumes `Xn` is already a distance matrix.
3. Initializes `Δ` as a zero matrix, setting diagonal elements to `w`.
4. Iterates over all pairs `(i, j)` to compute weighted filtration values using:
   ```julia
   weighted_filtration_value(D[i, j], w[i], w[j], p)

Converts Δ to a symmetric matrix and applies ripserer for persistence computation.

# Example
```julia
Xn = rand(10, 2)  # 10 points in 2D
w = rand(10)       # Random weights
wrips(Xn, w=w, p=2)  # Compute weighted Rips filtration
```
"""
function wrips(Xn; w=nothing, ρ=Euclidean(1e-12), p=1, type="points", args...)

    if type == "distMatrix"
        D = Xn
    else
        D = pairwise(ρ, Xn)
    end

    n_points = maximum(size(D))

    if isnothing(w)
        println("No weights specified. Taking W≡0")
        w = zeros(length(Xn))
    end

    Δ = zeros(size(D))
    Δ[diagind(Δ)] = w

    for i ∈ 1:n_points
        for j ∈ (i+1):n_points
            Δ[i, j] = weighted_filtration_value(D[i, j], w[i], w[j], p)
        end
    end

    return @pipe Δ |> Symmetric |> ripserer(_; args...)
end







# Clustering
"""
    extract_vertices(d, f=Ripserer.representative)

Extracts unique vertices from a persistence diagram using a specified representative function.

# Arguments
- `d`: A persistence diagram or filtration result.
- `f::Function=Ripserer.representative`: Function used to extract representative cycles.
  - Defaults to `Ripserer.representative`, which selects a representative cycle for each homology class.

# Returns
- A unique matrix of extracted vertices.

# Description
1. Applies the function `f` (default: `Ripserer.representative`) to `d`.
2. Extracts vertices from the representative cycles using `Ripserer.vertices`.
3. Converts them into an array of tuples.
4. Converts the array of tuples into a matrix using `wRips._ArrayOfTuples_to_Matrix`.
5. Returns a matrix of unique vertices.

# Example
```julia
d = ripserer(rand(10, 2))  # Compute persistent homology
V = extract_vertices(d)     # Extract unique vertices
```
"""
function extract_vertices(d, f=Ripserer.representative)
    return @pipe d |> f .|> Ripserer.vertices |> map(x -> [x...], _) |> wRips._ArrayOfTuples_to_Matrix |> unique
end;

"""
    stdscore(w)

Computes the standardized score (Z-score) for each element in `w`.

# Arguments
- `w::AbstractVector`: A vector of numerical values.

# Returns
- A vector of standardized scores, computed as:
  
  ```math
  z_i = \frac{|w_i - \text{mean}(w)|}{\text{std}(w)}

# Description
The standardized score (also known as the Z-score) measures how far each value in `w` deviates from the mean in terms of standard deviations. It is particularly useful for detecting outliers.

# Example
```julia
w = [1.0, 2.0, 3.0, 4.0, 5.0]
z_scores = stdscore(w)  # Computes standardized scores
```
"""
function stdscore(w)
    return [abs(x - mean(w)) / std(w) for x in w]
end;

"""
    filterDgm(dgm; order=1, ς=3, f=Ripserer.representative, vertex=true)

Filters a persistence diagram (`dgm`) based on standardized score deviation.

# Arguments
- `dgm`: A persistence diagram computed from `Ripserer.jl`.
- `order::Int=1`: The homology dimension to analyze (e.g., 0 for H₀, 1 for H₁, etc.).
- `ς::Real=3`: The threshold for filtering based on standard deviation.
- `f::Function=Ripserer.representative`: The function to extract representative cycles.
- `vertex::Bool=true`: Whether to return the extracted vertices (if `true`) or just their indices (if `false`).

# Returns
- If `vertex=true`: A list of filtered representative cycles as vertex sets.
- If `vertex=false`: The indices of significant persistence points.

# Description
This function filters features in the persistence diagram based on their standardized score (`stdscore`). 
It identifies features whose persistence deviates more than `ς` standard deviations from the mean. 
These features are considered significant and are either returned as vertex sets or their corresponding indices.

# Example
```julia
using Ripserer

# Compute persistence diagram
dgm = ripserer(rand(2, 50))

# Filter significant features in H₁
filtered_features = filterDgm(dgm, order=2, ς=2.5)
```
"""
function filterDgm(dgm; order=1, ς=3, f=Ripserer.representative, vertex=true)
    u = Ripserer.persistence.(dgm[order][1:end-1])
    v = [0, u[1:end-1]...]
    w = u - v
    index = findall(x -> x > ς, stdscore(w))
    return vertex ? [extract_vertices(dgm[order][i...], f) for i in index] : index
end;

"""
    dgmclust(dgm; order=1, threshold=3, n=nothing)

Clusters points in a dataset based on persistent homology features from a persistence diagram.

# Arguments
- `dgm`: A persistence diagram computed using `Ripserer.jl`.
- `order::Int=1`: The homology dimension to analyze (e.g., 0 for H₀, 1 for H₁, etc.).
- `threshold::Real=3`: The threshold for filtering features based on standard deviation.
- `n::Union{Int,Nothing}=nothing`: The total number of points in the dataset (defaults to `length(dgm[1])` if not provided).

# Returns
- `classes::Vector{Int}`: A vector of cluster assignments, where each point is assigned to a cluster 
  based on significant persistence features.

# Description
This function applies clustering based on the persistence diagram, identifying significant features 
using `filterDgm`. Each significant feature is assigned a unique cluster label, while all other points 
are assigned to cluster 0 (unclassified). 

# Example
```julia
using Ripserer

# Compute persistence diagram
dgm = ripserer(rand(2, 50))

# Cluster points based on H₁ features
clusters = dgmclust(dgm, order=1, threshold=2.5)
```
"""
function dgmclust(dgm; order=1, threshold=3, n=nothing)
    idx = filterDgm(dgm, order=order, ς=threshold)
    K = length(idx)
    if isnothing(n)
        n = length(dgm[1])
    end
    classes = repeat([0], n)
    for k in 1:K
        classes[idx[k]] .= k
    end
    return classes
end;