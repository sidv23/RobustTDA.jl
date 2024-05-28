############################################################
# k-PDTM
# 
# The following code is an adaptation of the k-PDTM algorithm in Julia.
# The original code was written in Python by Claire Brecheteau and is available at:
# https://github.com/GUDHI/TDA-tutorial/blob/master/Tuto-GUDHI-kPDTM-kPLM.ipynb
# 

function mean_var(X, x, q, kdt)
    NN = knn(kdt, x', q)[1]
    means = vcat(mean.([X[nn, :] for nn in NN], dims=1)...)
    vars = sum.(var.([X[nn, :] for nn in NN], dims=1, corrected=false))
    return means, vars
end

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