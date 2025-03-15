_Matrix_to_ArrayOfTuples = M -> Tuple.(eachcol(M)...)
_ArrayOfTuples_to_Matrix = A -> hcat(collect.(A)...)'
_ArrayOfVectors_to_ArrayOfTuples = A -> Tuple.(A)
_ArrayOfTuples_to_ArrayOfVectors = A -> [[a...] for a in A]


import Base: log, *

function Base.:*(a::Real, b::Tuple{Vararg{<:Real}})
    return a .* b
end

"""
    partition(x; n = nothing, e = nothing, kludge = 0.2, shift = 0)

Generates a partitioned range of values from the input data `x`.

# Arguments
- `x`: A collection of numerical values to be partitioned.
- `n`: (Optional) The number of partition points. Either `n` or `e` must be specified.
- `e`: (Optional) The step size between partition points. Either `n` or `e` must be specified.
- `kludge`: (Optional) A scaling factor applied to the upper bound to slightly extend the range (default: 0.2).
- `shift`: (Optional) A constant added to the upper bound for additional range extension (default: 0).

# Returns
- A range object spanning from the minimum value of `x` to an extended maximum, partitioned according to `n` or `e`.

# Description
This function partitions the values in `x` into a range that extends slightly beyond the maximum value. The extension is controlled by the `kludge` and `shift` parameters. The user must specify either `n` (number of partitions) or `e` (step size). If neither is provided, the function defaults to `n = 10`.

# Example Usage
```julia
x = [1.0, 2.5, 3.8, 5.0]
range1 = partition(x, n=5)  # Creates 5 partition points
range2 = partition(x, e=0.5)  # Creates partitions with step size 0.5
```
"""
function partition(x; n=nothing, e=nothing, kludge=0.2, shift=0)
    M = (minimum(x), maximum(x)) .* (1, 1 + kludge) .+ (0, shift)
    if !isnothing(n)
        return range(M[1], M[2], length=n)
    elseif !isnothing(e)
        return range(M[1], M[2], step=e)
    else
        println("Error: Either n or e must be specified. Defaulting to n=10")
        return partition(x, n=10)
    end
end

# blank = title -> plot(title=title, grid = false, showaxis = false, bottom_margin = -50Plots.px)
blank = (title; padding = -10) -> plot(title=title, framestyle=nothing, grid=false, showaxis=false, xticks=false, yticks=false, margin=padding * Plots.px)

"""
    vscatter(x; kwargs...)

Creates a scatter plot of the given vector of points.

# Arguments
- `x`: A vector of points, where each element is a coordinate tuple or a vector.
- `kwargs...`: Additional keyword arguments to customize the scatter plot.

# Returns
- A scatter plot visualization of `x`.

# Description
This function acts as a wrapper around `scatter`, converting the input vector `x` into a tuple format before plotting. It passes any additional keyword arguments to `scatter`, allowing customization of the plot.

# Example Usage
```julia
using Plots
x = [[1, 2], [3, 4], [5, 6]]
vscatter(x, markersize=5, color=:blue)
```
"""
function vscatter(x; kwargs...)
    scatter(Tuple.(x); kwargs...)
end

"""
    vscatter!(x; kwargs...)

Adds a scatter plot of the given vector of points to an existing plot.

# Arguments
- `x`: A vector of points, where each element is a coordinate tuple or a vector.
- `kwargs...`: Additional keyword arguments to customize the scatter plot.

# Returns
- Modifies the existing plot by adding a scatter plot of `x`.

# Description
This function acts as a wrapper around `scatter!`, converting the input vector `x` into a tuple format before plotting. It passes any additional keyword arguments to `scatter!`, allowing customization of the plot while overlaying it on an existing figure.

# Example Usage
```julia
using Plots
x = [[1, 2], [3, 4], [5, 6]]
scatter(x, markersize=5, color=:blue)  # Base scatter plot
vscatter!(x, markersize=8, color=:red) # Overlay additional scatter points
```
"""
function vscatter!(x; kwargs...)
    scatter!(Tuple.(x); kwargs...)
end

"""
    vplot(x; kwargs...)

Creates a plot of the given vector of points.

# Arguments
- `x`: A vector of points, where each element is a coordinate tuple or a vector.
- `kwargs...`: Additional keyword arguments to customize the plot.

# Returns
- A plot visualization of `x`.

# Description
This function acts as a wrapper around `plot`, converting the input vector `x` into a tuple format before plotting. It passes any additional keyword arguments to `plot`, allowing customization of the plot.

# Example Usage
```julia
using Plots
x = [[1, 2], [3, 4], [5, 6]]
vplot(x, linewidth=2, color=:red)
```
"""
function vplot(x; kwargs...)
    plot(Tuple.(x); kwargs...)
end

"""
    vplot!(x; kwargs...)

Adds a plot of the given vector of points to an existing plot.

# Arguments
- `x`: A vector of points, where each element is a coordinate tuple or a vector.
- `kwargs...`: Additional keyword arguments to customize the plot.

# Returns
- Modifies the existing plot by adding a new plot of `x`.

# Description
This function acts as a wrapper around `plot!`, converting the input vector `x` into a tuple format before plotting. It passes any additional keyword arguments to `plot!`, allowing customization of the plot while overlaying it on an existing figure.

# Example Usage
```julia
using Plots
x = [[1, 2], [3, 4], [5, 6]]
vplot(x, linewidth=2, color=:red)  # Base plot
vplot!(x, linewidth=1, color=:blue, linestyle=:dash) # Overlay additional plot
```
"""
function vplot!(x; kwargs...)
    plot!(Tuple.(x); kwargs...)
end