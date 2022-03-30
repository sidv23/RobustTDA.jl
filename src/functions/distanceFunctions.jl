#############################################################
############ MoM Distance

function dist(
    data::AbstractVector{T}
) where {T<:Union{Tuple{Vararg{<:Real}},Vector{<:Real}}}

    Xq = [data]

    trees = [KDTree(reduce(hcat, xq), leafsize = 1) for xq in Xq]

    return DistanceFunction(
        k = 1,
        trees = trees,
        X = Xq,
        type = "dist",
        Q = 1
    )
end



#############################################################
############ DTM


function dtm(
    data::AbstractVector{T},
    m::Real
) where {T<:Union{Tuple{Vararg{<:Real}},Vector{<:Real}}}

    tree = BruteTree(reduce(hcat, data), leafsize = 1)

    return DistanceFunction(
        k = floor(Int, m * length(data)),
        trees = [tree],
        X = [data],
        type = "dtm",
        Q = 1
    )
end


#############################################################
############ MoM DTM


function momdtm(
    data::AbstractVector{T},
    m::Real,
    Q::Integer = 0
) where {T<:Union{Tuple{Vararg{<:Real}},Vector{<:Real}}}

    if Q < 1
        println("Invalid value of Q supplied. Defaulting to Q = n_obs / 5 = $Q")
        Q = ceil(Int16, length(data) / 5)
    end

    Xq = [fold[2] for fold in kfolds(shuffleobs(data), Q)]

    trees = [KDTree(reduce(hcat, xq), leafsize = 1) for xq in Xq]

    return DistanceFunction(
        k = floor(Int, m * length(data)),
        trees = trees,
        X = Xq,
        type = "momdtm",
        Q = Q
    )
end


#############################################################
############ MoM Distance

function momdist(
    data::AbstractVector{T},
    Q = 0
) where {T<:Union{Tuple{Vararg{<:Real}},Vector{<:Real}}}

    if Q < 1
        println("Invalid value of Q supplied. Defaulting to Q = n_obs / 5 = $Q")
        Q = ceil(Int16, length(data) / 5)
    end

    Xq = [fold[2] for fold in kfolds(shuffleobs(data), Q)]

    trees = [KDTree(reduce(hcat, xq), leafsize = 1) for xq in Xq]

    return DistanceFunction(
        k = 1,
        trees = trees,
        X = Xq,
        type = "momdist",
        Q = Q
    )
end


#############################################################
############ Fit for Distance Functions


function fit(
    x::AbstractVecOrMat,
    D::DistanceFunction
)

    @unpack k, trees, X, type, Q = D

    if length(X[1]) < k
        k = ([ length(y) for y in X ] |> minimum)
    end

    fit = []
    for j ∈ 1:length(x)
        push!(fit,
            [knn(trees[i], x[j], k)[2] |> maximum for i = 1:Q]
        )
    end
    return reduce(vcat, median.(fit))
end
