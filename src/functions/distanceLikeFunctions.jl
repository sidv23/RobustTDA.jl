############################################################
# k-PDTM
# 
# The following code is an adaptation of the k-PDTM algorithm in Julia.
# The original code was written in Python by Claire Brecheteau and is available at:
# https://github.com/GUDHI/TDA-tutorial/blob/master/Tuto-GUDHI-kPDTM-kPLM.ipynb
# 
"""
    mean_var(X, x, q, kdt)

Computes the mean and variance of nearest neighbors for a given query point.

# Arguments
- `X`: A matrix where each row represents a data point.
- `x`: A query point (or multiple query points) for which neighbors are found.
- `q`: The number of nearest neighbors to consider.
- `kdt`: A k-d tree (`KDTree`) built from `X` for efficient nearest-neighbor search.

# Returns
- `means`: A matrix where each row represents the mean of the `q` nearest neighbors for the corresponding query point.
- `vars`: A vector containing the sum of variances of the `q` nearest neighbors for each query point.

# Description
This function finds the `q` nearest neighbors of `x` using the provided k-d tree (`kdt`). It then computes the mean and variance of these neighbors for each query point and returns them as separate arrays.

# Example Usage
```julia
using NearestNeighbors
X = rand(100, 2)  # 100 points in 2D space
kdt = KDTree(X')
x_query = [0.5, 0.5]
q = 5
means, vars = mean_var(X, x_query, q, kdt)
println("Means: ", means)
println("Variances: ", vars)
```
"""
function mean_var(X, x, q, kdt)
    NN = knn(kdt, x', q)[1]
    means = vcat(mean.([X[nn, :] for nn in NN], dims=1)...)
    vars = sum.(var.([X[nn, :] for nn in NN], dims=1, corrected=false))
    return means, vars
end

"""
    optima_for_kPDTM(X, q, k, sig, iter_max=10, nstart=1)

Finds optimal cluster centers using the k-Point Density-Topological Mode (k-PDTM) algorithm.

# Arguments
- `X`: A matrix where each row represents a d-dimensional data point.
- `q`: The number of nearest neighbors to consider.
- `k`: The number of clusters to find.
- `sig`: The number of points trimmed in the optimization process.
- `iter_max`: (Optional) Maximum number of iterations (default: 10).
- `nstart`: (Optional) Number of random initializations (default: 1).

# Returns
- `centers`: The optimized cluster centers.
- `means`: Mean values of the clusters based on k-nearest neighbors.
- `variances`: Variance values for each cluster.
- `colors`: Cluster assignments for each data point.
- `cost`: The final optimized cost value.

# Description
This function optimizes `k` cluster centers by iteratively assigning data points to clusters, updating centers, and trimming outliers. It maintains the best clustering result based on the cost function.

The process involves:
1. **Initialization**: Randomly selecting `k` centers.
2. **Iteration**: Updating cluster assignments and refining centers using nearest-neighbor statistics.
3. **Trimming**: Excluding `sig` points with the highest distance to ensure robustness.
4. **Optimization**: Keeping track of the best clustering configuration.

# Example Usage
```julia
X = rand(100, 2)  # 100 points in 2D space
q = 5
k = 3
sig = 10
centers, means, variances, colors, cost = optima_for_kPDTM(X, q, k, sig)
println("Optimized Centers: ", centers)
```
"""
function optima_for_kPDTM(X, q, k, sig, iter_max=10, nstart=1)
    n, d = size(X)
    opt_cost = Inf
    opt_centers = zeros(k, d)
    opt_colors = zeros(n)
    opt_kept_centers = zeros(Bool, k)
    costt, colors, min_distance, kept_centers, centers, old_centers, mv, Nstep = fill(nothing, 8)
    opt_mv = nothing


    if q <= 0 || q > n
        throw(ArgumentError("q should be in {1,2,...,n}"))
    elseif k <= 0 || k > n
        throw(ArgumentError("k should be in {1,2,...,n}"))
    end

    kdt = KDTree(permutedims(X))
    for start in 1:nstart
        colors = zeros(n)
        min_distance = zeros(n)
        kept_centers = trues(k)
        centers = X[randperm(n)[1:k], :]
        old_centers = fill(Inf, k, d)
        mv = mean_var(X, centers, q, kdt)
        Nstep = 1

        while any(old_centers .!= centers) && Nstep <= iter_max
            Nstep += 1
            # Update colors and min_distance
            for j in 1:n
                distances = sum((X[j, :]' .- mv[1][kept_centers, :]) .^ 2, dims=2) .+ mv[2][kept_centers]
                best_among_kept = argmin(distances)
                min_distance[j] = distances[best_among_kept]
                colors[j] = findall(kept_centers)[best_among_kept]
            end
            # Trimming step
            index = sortperm(min_distance, rev=true)
            colors[index[1:n-sig]] .= -1
            ds = min_distance[index[n-sig+1:end]]
            costt = mean(ds)

            # Update Centers and mv
            old_centers .= centers
            for i in findall(kept_centers)
                color_i = colors .== i
                if any(color_i)
                    centers[i, :] = mean(X[color_i, :], dims=1)
                else
                    kept_centers[i] = false
                end
            end
            mv = mean_var(X, centers, q, kdt)
        end

        if costt <= opt_cost
            opt_cost = costt
            opt_centers .= centers
            opt_mv = mv
            opt_colors .= colors
            opt_kept_centers .= kept_centers
        end
    end

    centers = opt_centers[kept_centers, :]
    means = opt_mv[1][kept_centers, :]
    variances = opt_mv[2][kept_centers]
    colors = zeros(n)
    for i in 1:n
        colors[i] = sum(kept_centers[1:min(k, round(Int, opt_colors[i] + 1))]) - 1
    end
    cost = opt_cost

    return centers, means, variances, colors, cost
end

"""
    kPDTM(X, query_pts, q, k, sig, iter_max=10, nstart=1)

Computes the k-Point Density-Topological Mode (k-PDTM) function for given query points.

# Arguments
- `X`: A vector of data points, where each element is a coordinate tuple or vector.
- `query_pts`: A vector of query points to evaluate the k-PDTM function.
- `q`: The number of nearest neighbors to consider.
- `k`: The number of clusters to form.
- `sig`: The number of points trimmed during optimization.
- `iter_max`: (Optional) Maximum number of iterations for optimization (default: 10).
- `nstart`: (Optional) Number of random initializations (default: 1).

# Returns
- `kPDTM_result`: The computed k-PDTM values for the given query points.
- `centers`: The optimized cluster centers.
- `means`: The mean values of the clusters.
- `variances`: The variance values for each cluster.
- `colors`: The final cluster assignments.
- `cost`: The final optimized cost value.

# Description
This function applies the k-PDTM algorithm to a dataset `X` and evaluates the resulting topological density structure at specified query points. It:
1. Validates input dimensions and parameters.
2. Calls `optima_for_kPDTM` to find optimal cluster centers.
3. Computes the k-PDTM function as the minimum squared distance to cluster means, incorporating variance.

# Example Usage
```julia
X = [rand(2) for _ in 1:100]  # 100 points in 2D space
query_pts = [rand(2) for _ in 1:10]  # 10 query points
q = 5
k = 3
sig = 10
kPDTM_result, centers, means, variances, colors, cost = kPDTM(X, query_pts, q, k, sig)
println("kPDTM Values: ", kPDTM_result)
```
"""
function kPDTM(X, query_pts, q, k, sig, iter_max=10, nstart=1)
    X = permutedims(hcat(X...))
    query_pts = permutedims(hcat(query_pts...))
    n, dx = size(X)
    _, dq = size(query_pts)

    if q <= 0 || q > n
        throw(ArgumentError("q should be in {1,2,...,n}"))
    elseif k <= 0 || k > n
        throw(ArgumentError("k should be in {1,2,...,n}"))
    elseif dx != dq
        throw(ArgumentError("X and query_pts should contain points with the same number of coordinates."))
    end

    centers, means, variances, colors, cost = optima_for_kPDTM(X, q, k, sig, iter_max, nstart)
    kPDTM_result = [minimum(sum((reshape(X, (:, 1, dx)) .- reshape(means, (1, :, dx))) .^ 2, dims=3) .+ reshape(variances, (1, :)), dims=2)...] .|> sqrt


    return kPDTM_result, centers, means, variances, colors, cost
end