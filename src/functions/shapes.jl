using Random, Distributions


############################################################
# LEMNISCATE
"""
    randLemniscate(n; sigma=0)

Generates `n` random points along a lemniscate-shaped curve with optional Gaussian noise.

# Arguments
- `n::Int`: The number of points to generate.
- `sigma::Real=0`: The standard deviation of the added Gaussian noise.

# Returns
- `points::Matrix{Float64}`: An `n × 2` matrix where each row represents a 2D point on the perturbed lemniscate.

# Description
This function constructs a lemniscate using the parametric equations:

    x = cos(t)
    y = sin(2t)

where `t` is linearly spaced from `0` to `2π`. Gaussian noise with standard deviation `sigma` is added to simulate variability.

# Example
```julia
n = 500
sigma = 0.05
points = randLemniscate(n, sigma=sigma)
```
"""
function randLemniscate(n; sigma=0)
    t = range(0, 2π, length=n)
    signal = hcat(cos.(t), sin.(2t)) .+ randn(n, 2) .* sigma
    noise = randn(n, 2) .* sigma
    return signal .+ noise
end

############################################################
# CIRCLE
"""
    randCircle(n::Int; sigma=0)

Generates `n` random points uniformly distributed on a unit circle with optional Gaussian noise.

# Arguments
- `n::Int`: The number of points to generate.
- `sigma::Real=0`: The standard deviation of the added Gaussian noise.

# Returns
- `points::Matrix{Float64}`: An `n × 2` matrix where each row represents a 2D point on the perturbed unit circle.

# Description
This function first generates `n` points sampled from a standard normal distribution in 2D space.
Each point is then normalized to lie on the unit circle. Optionally, Gaussian noise with standard deviation `sigma` is added to perturb the points.

# Example
```julia
n = 500
sigma = 0.05
points = randCircle(n, sigma=sigma)
```
"""
function randCircle(n::Int; sigma=0)
    signal = randn(n, 2) |> (x -> x ./ norm.(eachrow(x)))
    noise = randn(n, 2) .* sigma
    return signal .+ noise
end


############################################################
# UNIFORM DISTRIBUTION
"""
    randUnif(n::Int; a=0, b=1, d=2)

Generates `n` random points uniformly distributed in a `d`-dimensional space.

# Arguments
- `n::Int`: The number of points to generate.
- `a::Real=0`: The lower bound of the uniform distribution.
- `b::Real=1`: The upper bound of the uniform distribution.
- `d::Int=2`: The number of dimensions for each generated point.

# Returns
- `Matrix{Float64}`: An `n × d` matrix where each row represents a `d`-dimensional point sampled from a uniform distribution over `[a, b]`.

# Description
This function generates `n` points, each with `d` dimensions, by sampling from a uniform distribution `Uniform(a, b)`.

# Example
```julia
n = 1000
a, b = -1, 1
d = 2
points = randUnif(n, a=a, b=b, d=d)
```
"""
function randUnif(n::Int; a=0, b=1, d=2)
    return rand(Uniform(a, b), n, d)
end


############################################################
# MATÉRN CLUSTER
# 
# The following code is an adaptation of H. Paul Keeler's implementation of the
# Matérn cluster process in Julia. All credit goes to Paul.
# See: 
#   https://hpaulkeeler.com/simulating-a-matern-cluster-point-process/
#   https://github.com/hpaulkeeler/posts/blob/master/TestingJulia/MaternClusterRectangle.jl
# 
"""
    randMClust(n; a=1, b=1, λ_parent=5, λ_child=5, r=0.1)

Generates `n` points from a Matérn cluster process within a rectangular region.

# Arguments
- `n::Int`: The number of points to sample.
- `a::Real=1`: The width of the rectangular region.
- `b::Real=1`: The height of the rectangular region.
- `λ_parent::Real=5`: The intensity (expected number per unit area) of parent points.
- `λ_child::Real=5`: The expected number of child points per parent.
- `r::Real=0.1`: The cluster radius, defining the spread of child points around each parent.

# Returns
- `Matrix{Float64}`: An `n × 2` matrix where each row represents a sampled point.

# Description
This function simulates a **Matérn cluster process**, a spatial point process where:
1. Parent points are generated via a **Poisson process** with intensity `λ_parent` over an extended area.
2. Each parent generates a random number of child points following a **Poisson distribution** with mean `λ_child`.
3. Child points are placed around the parent following a **radial uniform distribution** within the cluster radius `r`.
4. A subset of `n` points is randomly selected from the generated child points.

# Example
```julia
n = 1000
a, b = 2, 2
λ_parent, λ_child, r = 5, 10, 0.2
points = randMClust(n, a=a, b=b, λ_parent=λ_parent, λ_child=λ_child, r=r)
```
"""
function randMClust(n; a=1, b=1, λ_parent=5, λ_child=5, r=0.1)
    # Generate n points from a Matérn cluster process with intensity λ_parent and λ_child
    # in a rectangle of dimensions a x b with cluster radius r.
    vol = 4 * (a + r) * (b + r)
    N_Parent = 1 + rand(Poisson(λ_parent * vol))
    N_child = [1 + rand(Poisson(λ_child)) for i in 1:N_Parent]
    rand_indx = rand(1:N_Parent)
    N = sum(N_child)
    θ = 2π * rand(N)
    R = r .* sqrt.(rand(N))
    parent_coords = rand(Uniform(-1, 1), N_Parent, 2) .* [a + r b + r]
    X_parent = fill.(eachrow(parent_coords), N_child) |> (x -> permutedims(hcat(vcat(x...)...)))
    X_child = ([cos.(θ) sin.(θ)] .* R) .+ X_parent
    Xn = permutedims(hcat(rand(eachrow(X_child), n)...))
    return Xn
end

############################################################
# sample points from the line y = mx + c from x_min to x_max
"""
    randLine(n::Int; m=1, c=0, x_min=0, x_max=1)

Generates `n` random points along a straight line `y = mx + c` with uniformly distributed `x` values.

# Arguments
- `n::Int`: Number of points to generate.
- `m::Real=1`: Slope of the line.
- `c::Real=0`: Intercept of the line.
- `x_min::Real=0`: Minimum value of `x`.
- `x_max::Real=1`: Maximum value of `x`.

# Returns
- `Matrix{Float64}`: An `n × 2` matrix where each row represents a `(x, y)` point on the line.

# Description
This function generates `n` points where the `x` values are uniformly sampled from `[x_min, x_max]`, and the corresponding `y` values are computed using the equation `y = mx + c`.

# Example
```julia
n = 100
x_min = -5
x_max = 5

points = randLine(n, m=1, c=0, x_min=x_min, x_max=x_max)
```
"""
function randLine(n::Int; m=1, c=0, x_min=0, x_max=1)
    x = rand(Uniform(x_min, x_max), n)
    y = m .* x .+ c
    return hcat(x, y)
end