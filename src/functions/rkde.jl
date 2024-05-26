function med_phi(x)
    return (((x .> 0) .* 2) .- 1) .* (1 ./ x)
end

function med_rho(x)
    return abs.(x)
end

function hampel_rho(x; a, b, c)
    return ( 
    x -> 0 ≤ x < a ? 0.5 * x^2 : 
    a ≤ x < b ? ( a * x ) - ( 0.5 * (a^2) ) :
    b ≤ x < c ? ( ( 0.5 * a  / ( b - c) ) * ( (x - c)^2 ) ) + ( 0.5 * a * ( b + c -a ) ) :
    c ≤ x ? ( 0.5 * a * ( b + c -a ) ) :
    error("Invalid value of x")
    ).(x)
end

function hampel_phi(x; a, b, c)
    return (
    x -> 0 ≤ x < a ? 1 :
    a ≤ x < b ? a / x :
    b ≤ x < c ? ( a * ( c - x ) ) / ( x * (c - b) ) :
    c ≤ x ? 0 :
    error("Invalid value of x")
    ).(x)
end

function rkhs_norm(X, k::Kernel, w = nothing)
    
    n = length(X)

    if isnothing(w)
        w = repeat([1], n) ./ n
    end

    Kxx = kernelmatrix(k, X)
    a = Kxx * w
    b = w'a
    norm = [ Kxx[i,  i] + b - ( 2 * a[i] ) for i in 1:n ]

    return sqrt.(norm)
end

function rkde_loss(X, k::Kernel, fun::T, w = nothing) where {T<:Function}
    return rkhs_norm(X, k, w) .|> fun |> sum
end


function rkde_w(w, X, k::Kernel; loss::T, ϕ, tolerance=1e-10, message=false) where {T<:Function}
    L_old = rkde_loss(X, k, loss)
    ratio = 1
    iter = 0

    while ((ratio > tolerance) && (iter < 200))
        
        w_phi = ϕ( rkhs_norm(X, k, w) )
        w = w_phi ./ sum( w_phi )
        
        L_new = rkde_loss(X, k, loss, w)
        ratio = abs( (L_new - L_old) ./ L_old )

        L_old = L_new
        iter = iter + 1

        if message
            println("iter = $iter, ratio=$ratio")
        end

    end

    return w
end


function rkde_W(X; k::Kernel)
    nx = length(X)
    w = rand(nx)
    w = w ./ sum(w)

    w_med = rkde_w(w, X, k; loss=med_rho, ϕ=med_phi)
    d = rkhs_norm(X, k, w_med)

    q = quantile(d, [0.5, 0.9, 0.95])
    a = q[1]
    b = q[2]
    c = q[3]

    H_ρ = x -> hampel_rho(x, a=a, b=b, c=c)
    H_ϕ = x -> hampel_phi(x; a=a, b=b, c=c)

    w_new = rkde_w(w_med, X, k; loss=H_ρ, ϕ=H_ϕ)

    return w_new
end

# function rkde(G; X, k::Kernel)
#     w = rkde_W(X; k = k)
#     return [(kernelmatrix(k, [g], X)*w)[1] for g in G]
# end


function fit(x; X, w, k::Kernel)
    dists = sqrt.(k(0, 0) .- sum(kernelmatrix(k, x, X) .* w', dims=2))
    return [dists...]
end

function rkde_fit(G; X, w, k::Kernel)
    return [(kernelmatrix(k, [g], X)*w)[1] for g in G]
end

