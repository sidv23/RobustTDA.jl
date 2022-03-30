_Matrix_to_ArrayOfTuples = M -> Tuple.(eachcol(M)...)
_ArrayOfTuples_to_Matrix = A -> hcat(collect.(A)...)'
_ArrayOfVectors_to_ArrayOfTuples = A -> Tuple.(A)
_ArrayOfTuples_to_ArrayOfVectors = A -> [[a...] for a in A]


import Base: log, *

function Base.:*(a::Real, b::Tuple{Vararg{<:Real}})
    return a .* b
end


function partition(x; n = nothing, e = nothing, kludge = 0.2, shift = 0)
    M = (minimum(x), maximum(x)) .* (1, 1 + kludge) .+ (0, shift)
    if !isnothing(n)
        return range(M[1], M[2], length = n)
    elseif !isnothing(e)
        return range(M[1], M[2], step = e)
    else
        println("Error: Either n or e must be specified. Defaulting to n=10")
        return partition(x, n = 10)
    end
end

# blank = title -> plot(title=title, grid = false, showaxis = false, bottom_margin = -50Plots.px)
blank = (title; padding = -10) -> plot(title = title, framestyle = nothing, grid = false, showaxis = false, xticks = false, yticks = false, margin = padding * Plots.px)

function vscatter(x; kwargs...)
    scatter(Tuple.(x); kwargs...)
end

function vscatter!(x; kwargs...)
    scatter!(Tuple.(x); kwargs...)
end

function vplot(x; kwargs...)
    plot(Tuple.(x); kwargs...)
end

function vplot!(x; kwargs...)
    plot!(Tuple.(x); kwargs...)
end
