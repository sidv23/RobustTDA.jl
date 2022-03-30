@with_kw mutable struct plot_params
    col::String = "orange"
    alpha::Float16 = 0.01
    lwd::Float16 = 0
    lalpha::Float16 = 0.0
    lcol::String = col
    msize::Float16 = 1
end

@with_kw mutable struct lepski_params
    a::Number = 0.5
    b::Number = 1.0
    mmin::Integer = 2
    mmax::Integer = 200
    pi::Number = 1.1
    δ::Float16 = 0.05
end


@with_kw mutable struct RKHS
    k::Kernel
    kinv::Function
end

@with_kw mutable struct wRips_params
    f::Any
    p::Number = 1
    ρ::Any
end

@with_kw mutable struct DistanceLikeFunction
    k::Kernel
    σ::Vector{T} where {T<:Real}
    X::AbstractVector{<:AbstractVector{<:Union{Tuple{Vararg{<:Real}},Vector{<:Real}}}}
    Kxx::AbstractVecOrMat
    type::String
    Q::Int64
end

@with_kw mutable struct DistanceFunction
    k::Integer
    trees::AbstractVector{<:NNTree}
    X::AbstractVector{<:AbstractVector{<:Union{Tuple{Vararg{<:Real}},Vector{<:Real}}}}
    type::String
    Q::Int64
end
