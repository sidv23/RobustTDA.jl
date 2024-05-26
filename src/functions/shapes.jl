using Random, Distributions


############################################################
# LEMNISCATE
function randLemniscate(n; sigma = 0)
    t = range(0, 2π, length = n)
    signal = hcat(cos.(t), sin.(2t)) .+ randn(n, 2) .* sigma
    noise = randn(n, 2) .* sigma
    return signal .+ noise
end

############################################################
# CIRCLE
function randCircle(n::Int; sigma = 0)
    signal = randn(n, 2)  |> (x -> x ./ norm.(eachrow(x)))
    noise = randn(n, 2) .* sigma
    return signal .+ noise
end


############################################################
# UNIFORM DISTRIBUTION
function randUnif(n::Int; a = 0, b = 1, d = 2)
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
