"""
    med_phi(x)

Computes a transformation of `x` using a piecewise function.

# Arguments
- `x`: A numeric input or array.

# Returns
- An array where each element is computed as `2/x` for positive `x` and `-1/x` for negative `x`.

# Description
This function applies the transformation:
  - `2/x` if `x > 0`
  - `-1/x` if `x <= 0`

# Example Usage
```julia
x = [-2, -1, 0, 1, 2]
med_phi(x)
```
"""
function med_phi(x)
    return (((x .> 0) .* 2) .- 1) .* (1 ./ x)
end

"""
    med_rho(x)

Computes the absolute value of the input.

# Arguments
- `x`: A numeric input or array.

# Returns
- An array where each element is the absolute value of the corresponding element in `x`.

# Description
This function applies the absolute value operation element-wise to `x`.

# Example Usage
```julia
x = [-3, -1, 0, 2, 5]
med_rho(x)  # Returns [3, 1, 0, 2, 5]
```
"""
function med_rho(x)
    return abs.(x)
end

"""
    hampel_rho(x; a, b, c)

Applies the Hampel rho function to the input `x`, using parameters `a`, `b`, and `c`.

# Arguments
- `x`: A numeric input or array.
- `a`, `b`, `c`: Threshold parameters defining different regions of the function.

# Returns
- An array where each element is transformed according to the Hampel rho function:
  - `0.5 * x^2` if `0 ≤ x < a`
  - `(a * x) - (0.5 * a^2)` if `a ≤ x < b`
  - `((0.5 * a / (b - c)) * ((x - c)^2)) + (0.5 * a * (b + c - a))` if `b ≤ x < c`
  - `0.5 * a * (b + c - a)` if `x ≥ c`
  - Throws an error for invalid values of `x`.

# Example Usage
```julia
x = [-2, 0, 1, 3, 5]
hampel_rho(x; a=1, b=3, c=5)
```
"""
function hampel_rho(x; a, b, c)
    return (
        x -> 0 ≤ x < a ? 0.5 * x^2 :
             a ≤ x < b ? (a * x) - (0.5 * (a^2)) :
             b ≤ x < c ? ((0.5 * a / (b - c)) * ((x - c)^2)) + (0.5 * a * (b + c - a)) :
             c ≤ x ? (0.5 * a * (b + c - a)) :
             error("Invalid value of x")
    ).(x)
end

"""
    hampel_phi(x; a, b, c)

Applies the Hampel phi function to the input `x`, using parameters `a`, `b`, and `c`.

# Arguments
- `x`: A numeric input or array.
- `a`, `b`, `c`: Threshold parameters defining different regions of the function.

# Returns
- An array where each element is transformed according to the Hampel phi function:
  - `1` if `0 ≤ x < a`
  - `a / x` if `a ≤ x < b`
  - `(a * (c - x)) / (x * (c - b))` if `b ≤ x < c`
  - `0` if `x ≥ c`
  - Throws an error for invalid values of `x`.

# Example Usage
```julia
x = [-2, 0, 1, 3, 5]
hampel_phi(x; a=1, b=3, c=5)
```
"""
function hampel_phi(x; a, b, c)
    return (
        x -> 0 ≤ x < a ? 1 :
             a ≤ x < b ? a / x :
             b ≤ x < c ? (a * (c - x)) / (x * (c - b)) :
             c ≤ x ? 0 :
             error("Invalid value of x")
    ).(x)
end

"""
    rkhs_norm(X, k::Kernel, w=nothing)

Computes the RKHS (Reproducing Kernel Hilbert Space) norm for a set of points `X` using a specified kernel function `k`.

# Arguments
- `X`: A collection of data points.
- `k::Kernel`: A kernel function used to compute the RKHS norm.
- `w`: An optional weight vector. If `nothing`, uniform weights are used.

# Returns
- A vector containing the RKHS norm values for each data point in `X`.

# Example Usage
```julia
X = [[1.0], [2.0], [3.0]]
k = GaussianKernel(1.0)
rkhs_norm(X, k)
```
"""
function rkhs_norm(X, k::Kernel, w=nothing)

    n = length(X)

    if isnothing(w)
        w = repeat([1], n) ./ n
    end

    Kxx = kernelmatrix(k, X)
    a = Kxx * w
    b = w'a
    norm = [Kxx[i, i] + b - (2 * a[i]) for i in 1:n]

    return sqrt.(norm)
end

"""
    rkde_loss(X, k::Kernel, fun::T, w=nothing) where {T<:Function}

Computes the RKDE (Reproducing Kernel Density Estimator) loss for a given dataset `X`, kernel `k`, and transformation function `fun`.

# Arguments
- `X`: A collection of data points.
- `k::Kernel`: A kernel function used for density estimation.
- `fun::T where {T<:Function}`: A function applied to the RKHS norm values.
- `w`: An optional weight vector. If `nothing`, uniform weights are used.

# Returns
- A scalar representing the sum of transformed RKHS norm values.

# Example Usage
```julia
X = [[1.0], [2.0], [3.0]]
k = GaussianKernel(1.0)
fun = x -> x^2
rkde_loss(X, k, fun)
```
"""
function rkde_loss(X, k::Kernel, fun::T, w=nothing) where {T<:Function}
    return rkhs_norm(X, k, w) .|> fun |> sum
end

"""
    rkde_w(w, X, k::Kernel; loss::T, ϕ, tolerance=1e-10, message=false) where {T<:Function}

Computes the weighted density estimation using a reproducing kernel Hilbert space (RKHS) approach.

## Parameters:
- `w`: Initial weights, a vector of probabilities.
- `X`: Data points, a vector of vectors or tuples.
- `k`: Kernel function used for density estimation.
- `loss`: A function used to compute loss in the RKDE framework.
- `ϕ`: Influence function used to update the weights.
- `tolerance`: Convergence tolerance (default: `1e-10`).
- `message`: Boolean flag to print iteration details (default: `false`).

## Returns:
- Updated weights after optimization.

## Example Usage:
```julia
using KernelFunctions
k = GaussianKernel()
X = [[1.0], [2.0], [3.0]]
w = [1/3, 1/3, 1/3]
loss(x) = x^2
ϕ(x) = exp.(-x)
new_w = rkde_w(w, X, k; loss=loss, ϕ=ϕ)
```
"""
function rkde_w(w, X, k::Kernel; loss::T, ϕ, tolerance=1e-10, message=false) where {T<:Function}
    L_old = rkde_loss(X, k, loss)
    ratio = 1
    iter = 0

    while ((ratio > tolerance) && (iter < 200))

        w_phi = ϕ(rkhs_norm(X, k, w))
        w = w_phi ./ sum(w_phi)

        L_new = rkde_loss(X, k, loss, w)
        ratio = abs((L_new - L_old) ./ L_old)

        L_old = L_new
        iter = iter + 1

        if message
            println("iter = $iter, ratio=$ratio")
        end

    end

    return w
end

"""
    rkde_W(X; k::Kernel)

Computes robust kernel density estimation (RKDE) weights for a given dataset `X` using a specified kernel function `k`.

# Arguments
- `X::Vector{T}`: A dataset where each element is a point in the space.
- `k::Kernel`: A kernel function used for density estimation.

# Returns
- `w_new::Vector{Float64}`: The computed robust weights for the data points.

# Description
1. Initializes random weights for `X` and normalizes them.
2. Computes an initial robust weighting using the `med_rho` loss function and `med_phi` influence function.
3. Computes RKHS distances using the obtained weights.
4. Estimates threshold values (`a`, `b`, `c`) using the 50th, 90th, and 95th percentiles of RKHS distances.
5. Uses the Hampel loss (`hampel_rho`) and influence function (`hampel_phi`) to further refine the robust weights.

# Example
```julia
k = GaussianKernel(1.0)
X = [rand(2) for _ in 1:100]
weights = rkde_W(X, k=k)
```
"""
function rkde_W(X; k::Kernel)
    nx = length(X)
    w = rand(nx)
    w = w ./ sum(w)

    w_med = rkde_w(w, X, k; loss=med_rho, ϕ=med_phi)
    d = rkhs_norm(X, k, w_med)

    q = quantile(d, [0.5, 0.9, 0.95])
    a = q[1]
    b = q[2]
    c = q[3]

    H_ρ = x -> hampel_rho(x, a=a, b=b, c=c)
    H_ϕ = x -> hampel_phi(x; a=a, b=b, c=c)

    w_new = rkde_w(w_med, X, k; loss=H_ρ, ϕ=H_ϕ)

    return w_new
end

# function rkde(G; X, k::Kernel)
#     w = rkde_W(X; k = k)
#     return [(kernelmatrix(k, [g], X)*w)[1] for g in G]
# end

"""
    fit(x; X, w, k::Kernel)

Computes the distance between each point in `x` and the dataset `X` using a kernel function `k` and weight vector `w`.

# Arguments
- `x::Vector{T}`: A set of query points.
- `X::Vector{T}`: The reference dataset.
- `w::Vector{Float64}`: Weights associated with `X`.
- `k::Kernel`: A kernel function used to compute similarities.

# Returns
- `dists::Vector{Float64}`: A vector of computed distances for each point in `x`.

# Description
The function computes the RKHS-induced distance between `x` and `X` using the kernel function `k`. The calculation follows:
    
    dists = sqrt(k(0, 0) - sum(kernelmatrix(k, x, X) .* w', dims=2))

This measures the deviation of `x` from `X` under the given kernel, weighted by `w`.

# Example
```julia
k = GaussianKernel(1.0)
X = [rand(2) for _ in 1:100]
w = rand(length(X))
w /= sum(w)  # Normalize weights
x = [rand(2) for _ in 10]
distances = fit(x; X=X, w=w, k=k)
```
"""
function fit(x; X, w, k::Kernel)
    dists = sqrt.(k(0, 0) .- sum(kernelmatrix(k, x, X) .* w', dims=2))
    return [dists...]
end

"""
    rkde_fit(G; X, w, k::Kernel)

Computes the kernel density estimate (KDE) values at query points `G` based on a weighted dataset `X` and a kernel function `k`.

# Arguments
- `G::Vector{T}`: A set of query points where the KDE is evaluated.
- `X::Vector{T}`: The reference dataset used to estimate the density.
- `w::Vector{Float64}`: Weights associated with `X`, representing relative importance in density estimation.
- `k::Kernel`: A kernel function used to compute similarities.

# Returns
- `rkde_values::Vector{Float64}`: A vector containing the estimated density values at each point in `G`.

# Description
This function computes the KDE estimate for each query point `g ∈ G` using:

    rkde_value = (kernelmatrix(k, [g], X) * w)[1]

where `kernelmatrix(k, [g], X)` computes the kernel similarities between `g` and all points in `X`, and the result is weighted by `w`.

# Example
```julia
k = GaussianKernel(1.0)
X = [rand(2) for _ in 1:100]
w = rand(length(X))
w /= sum(w)  # Normalize weights
G = [rand(2) for _ in 10]
kde_values = rkde_fit(G; X=X, w=w, k=k)
```
"""
function rkde_fit(G; X, w, k::Kernel)
    return [(kernelmatrix(k, [g], X)*w)[1] for g in G]
end