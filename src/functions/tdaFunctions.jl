#############################################################
############ Balls

function bandwidth_select(Xn, k)

    if k < 2
        k = 2
    end

    kdtree = KDTree(reduce(hcat, Xn), leafsize = 1)
    knns = [knn(kdtree, Xn[i], k)[2] |> maximum for i = eachindex(Xn)]
    return median(knns)
end


#############################################################
############ Balls

function Balls(X, R)
    if length(R) == 1
        return [Ball2([x...], float(R)) for x in X]
    else
        return [Ball2([x...], float(r)) for (x, r) in zip(X, R)]
    end
end


#############################################################
############ Radius Function

function rfx(t, w, p)
    return t .> w ? (p != Inf ? (t^p .- w .^ p) .^ (1 / p) : t) : 0
end



#############################################################
############ Convert Distance

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

function wrips(Xn; w = nothing, ρ = Euclidean(1e-12), p = 1, type = "points", args...)

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

function extract_vertices(d, f = Ripserer.representative)
    return @pipe d |> f .|> Ripserer.vertices |> map(x -> [x...], _) |> wRips._ArrayOfTuples_to_Matrix |> unique
end;

function stdscore(w)
    return [abs(x - mean(w)) / std(w) for x in w]
end;

function filterDgm(dgm; order = 1, ς = 3, f = Ripserer.representative, vertex = true)
    u = Ripserer.persistence.(dgm[order][1:end-1])
    v = [0, u[1:end-1]...]
    w = u - v
    index = findall(x -> x > ς, stdscore(w))
    return vertex ? [extract_vertices(dgm[order][i...], f) for i in index] : index
end;

function dgmclust(dgm; order = 1, threshold = 3, n=nothing)
    idx = filterDgm(dgm, order = order, ς = threshold)
    K = length(idx)
    if isnothing(n)
        n = length(dgm[1])
    end
    classes = repeat([0],n)
    for k in 1:K
        classes[idx[k]] .= k
    end
    return classes
end;