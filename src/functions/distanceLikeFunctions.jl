#############################################################
############ Hilbertian Metric

function dH(k::Kernel)
    # return function(x,y) sqrt(k(x,x) + k(y,y) - 2k(x,y)) end
    return function (x, y)
        sqrt(2 - 2 * k(x, y))
    end
end

#############################################################
############ MMD

function MMD(X::AbstractVector, Y::AbstractVector; k::Kernel)
    m = length(X)
    n = length(Y)
    Kxx = kernelmatrix(k, X) |> sum
    Kyy = kernelmatrix(k, Y) |> sum
    Kxy = kernelmatrix(k, X, Y) |> sum
    return sqrt((Kxx / (m^2)) + (Kyy / (n^2)) - 2 * (Kxy / (m * n)))
end

#############################################################
############ Kernel Distance

function KDist(
    data::AbstractVector{T},
    k::Kernel
) where {T<:Union{Tuple{Vararg{<:Real}},Vector{<:Real}}}
    Kxx = kernelmatrix(k, data) |> Statistics.mean
    return DistanceLikeFunction(
        k = k,
        σ = :transform ∈ fieldnames(typeof(k)) ? k.transform.s : [1.0],
        X = [data],
        Kxx = [Kxx],
        type = "vanilla",
        Q = 1
    )
end

#############################################################
############ MoM Kernel Distance


function momKDist(
    data::AbstractVector{T},
    k::Kernel;
    Q::Integer = 0
) where {T<:Union{Tuple{Vararg{<:Real}},Vector{<:Real}}}

    if Q < 1
        println("Invalid value of Q supplied. Defaulting to Q = n_obs / 5 = $Q")
        Q = ceil(Int16, length(data) / 5)
    end

    Xq = [fold[2] for fold in kfolds(shuffleobs(data), Q)]

    Kxx = [(kernelmatrix(k, xq) |> Statistics.mean) for xq in Xq]

    return DistanceLikeFunction(
        k = k,
        σ = :transform ∈ fieldnames(typeof(k)) ? k.transform.s : [1.0],
        X = Xq,
        Kxx = Kxx,
        type = "mom",
        Q = Q
    )
end

#############################################################
############ Fit function for Distance Like Function


function fit(
    x::Union{T,AbstractVector{T}},
    D::DistanceLikeFunction
    ) where {T<:Union{Tuple{Vararg{<:Real}},Vector{<:Real}}}
    
    @unpack k, X, Kxx, Q = D
    
    if typeof(x) <: T
        x = [x]
    end
    
    fit = []
    for j ∈ 1:length(x)
        push!(fit,
        [sqrt(k(0, 0) + Kxx[i] - 2 * (kernelmatrix(k, X[i], [x[j]]) |> Statistics.mean)) for i = 1:Q]
        )
    end
    return reduce(vcat, median.(fit))
end
