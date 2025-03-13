"""
    plot_balls(B; X = nothing, p = nothing, par::plot_params)

Plots a set of balls and optionally overlays scatter points on an existing plot.

# Arguments
- `B`: The set of balls to be plotted.
- `X`: (Optional) A set of points to overlay as a scatter plot.
- `p`: (Optional) An existing plot object to overlay the visualization on.
- `par::plot_params`: A struct containing plotting parameters.

# Returns
- A plot object displaying the set of balls and optionally the scatter points.

# Description
This function visualizes a set of balls with specified graphical parameters. If `X` is provided, it overlays scatter points. If `p` is provided, the visualization is added to an existing plot.

# Example Usage
```julia
using Plots
B = [...]  # Define set of balls
X = rand(2, 10)  # Generate random points
params = plot_params(col=:blue, alpha=0.5, lwd=2, lalpha=0.8)
plot_balls(B, X=X, par=params)  # Plot balls with points
```
"""
function plot_balls(B; X=nothing, p=nothing, par::plot_params)
    if isnothing(X) & isnothing(p)
        p = plot(B,
            c=par.col, fillalpha=par.alpha, linewidth=par.lwd,
            linealpha=par.lalpha, linecolor=par.col, ratio=1)

    elseif isnothing(X)
        p = plot(B,
            c=par.col, fillalpha=par.alpha, linewidth=par.lwd,
            linealpha=par.lalpha, linecolor=par.col, ratio=1)

    elseif isnothing(p)
        p = scatter(X, ratio=1, label=nothing)
        p = plot(p, B,
            c=par.col, fillalpha=par.alpha, linewidth=par.lwd,
            linealpha=par.lalpha, linecolor=par.col, ratio=1)

    else
        p = scatter(p, X, ratio=1, label=nothing)
        p = plot(B,
            c=par.col, fillalpha=par.alpha, linewidth=par.lwd,
            linealpha=par.lalpha, linecolor=par.col, ratio=1)
    end

    return p
end


"""
    filtration_plot(t;
        Xn::Vector{<:Vector{<:Real}},
        w::Vector{<:Real}=nothing,
        p=1,
        par::plot_params,
        clim=nothing
    )

Generates a filtration plot of a given point cloud with radius function values.

# Arguments
- `t`: The filtration parameter controlling the radius of the plotted balls.
- `Xn`: A vector of points, where each point is represented as a vector of real numbers.
- `w`: (Optional) A vector of weight values associated with each point. Defaults to zero weights.
- `p`: (Optional) The power parameter for the radius function. Default is `1`.
- `par::plot_params`: A struct containing plotting parameters for visual customization.
- `clim`: (Optional) Color limits for the scatter plot based on weights. Defaults to the range of `w`.

# Returns
- A plot object displaying the filtration process.

# Description
This function visualizes a filtration process over a point cloud. It first plots the points using a scatter plot, then overlays the corresponding balls (using `plot_balls`), and finally re-plots the points to ensure proper visibility. The radius of the balls is determined by `rfx.(t, w, p)`.

# Example Usage
```julia
using Plots
Xn = [rand(2) for _ in 1:10]  # Generate random points
w = rand(10)  # Assign random weights
params = plot_params(col=:blue, alpha=0.3, lwd=1.5)
filtration_plot(0.5, Xn=Xn, w=w, par=params)
```
"""
function filtration_plot(t;
    Xn::Vector{<:Vector{<:Real}},
    w::Vector{<:Real}=nothing,
    p=1,
    par::plot_params,
    clim=nothing
)

    if isnothing(w)
        w = repeat([0], length(Xn))
    end

    if isnothing(clim)
        clim = extrema(w)
    end

    plt = scatter(Tuple.(Xn), marker_z=w, label=nothing, ratio=1, clim=clim)
    plt = plot_balls(Balls(Xn, rfx.(t, w, p)), p=plt, par=par)
    plt = scatter(plt, Tuple.(Xn), marker_z=w, label=nothing, ratio=1, clim=clim)

    return plt
end

"""
    surfacePlot(xseq::Any, yseq=nothing; f::Function, args...)

Generates a 3D surface plot for a given function over specified x and y sequences.

# Arguments
- `xseq::Any`: A sequence of x-values over which the function is evaluated.
- `yseq`: (Optional) A sequence of y-values. If not provided, `yseq` defaults to `xseq`.
- `f::Function`: The function to be plotted, taking two arguments (x, y).
- `args...`: Additional keyword arguments passed to the `plot` function.

# Returns
- A plot object displaying the 3D surface plot.

# Description
This function creates a surface plot of `f(x, y)` over the specified `xseq` and `yseq` ranges. If `yseq` is not provided, it is assumed to be the same as `xseq`. The function then uses the `plot` function with `st=:surface` to generate the 3D visualization and displays the result.

# Example Usage
```julia
using Plots
x = range(-2, 2, length=50)
y = range(-2, 2, length=50)
f(x, y) = sin(x) * cos(y)
surfacePlot(x, y; f=f)
```
"""
function surfacePlot(xseq::Any, yseq=nothing; f::Function, args...)
    if yseq |> isnothing
        yseq = xseq
    end
    plt = plot(xseq, yseq, f, st=:surface; args...)
    display(plt)
    return plt
end