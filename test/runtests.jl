using RobustTDA
import RobustTDA as rtda
using Ripserer, KernelFunctions
using Test

# Signal from a circle with noise
n, m = 500, 100
N = n + m
signal = 2 .* rtda.randCircle(n, sigma=0.05)
noise = rtda.randMClust(m, a=1, b=1, λ_parent=2, λ_child=100, r=0.1)
X = [signal; noise]
Xn = [[x...] for x in Tuple.(eachrow(X))]

# Distance-like Functions
f_dist, f_dtm, f_momdist = rtda.dist(Xn), dtm(Xn, 0.1), momdist(Xn, 2m + 1)

# Weights
w_dist = rtda.fit(Xn, f_dist)
w_dtm = rtda.fit(Xn, f_dtm)
w_momdist = rtda.fit(Xn, f_momdist)

# Vanilla Distance (Diagram)
D_dist = Ripserer.ripserer(Xn)

# Distance to Measure (Diagram)
D_dtm_1 = rtda.wrips(Xn, w=w_dtm, p=1)
D_dtm_2 = rtda.wrips(Xn, w=w_dtm, p=2)
D_dtm_inf = rtda.wrips(Xn, w=w_dtm, p=Inf)

# Median of Means Distance (Diagram)
D_momdist_1 = rtda.wrips(Xn, w=w_momdist, p=1)
D_momdist_2 = rtda.wrips(Xn, w=w_momdist, p=2)
D_momdist_inf = rtda.wrips(Xn, w=w_momdist, p=Inf)

@testset "RobustTDA.jl" begin
    @test typeof(f_dist) == rtda.DistanceFunction
    @test typeof(f_dtm) == rtda.DistanceFunction
    @test typeof(f_momdist) == rtda.DistanceFunction

    @test typeof(D_dist) <: Vector{Ripserer.PersistenceDiagram}
    @test typeof(D_dtm_2) <: Vector{Ripserer.PersistenceDiagram}
    @test typeof(D_momdist_inf) <: Vector{Ripserer.PersistenceDiagram}
end

@testset "Expected Behavior" begin
    @test length(w_dist) == N
    @test w_dist == zeros(N)
    @test length(w_dtm) == N
    @test length(w_momdist) == N

    # @test prod(
    #     length.(D_dtm_inf) .<=
    #     length.(D_dtm_2) .<=
    #     length.(D_dtm_1) .<=
    #     length.(D_dist)
    # )

    # @test prod(
    #     length.(D_momdist_inf) .<=
    #     length.(D_momdist_2) .<=
    #     length.(D_momdist_1) .<=
    #     length.(D_dist)
    # )
end

@testset "Autotune" begin
    θ = rtda.lepski_params(
        a=0.2,
        b=1,
        mmin=50,
        mmax=200,
        pi=1.1,
        δ=0.01
    )
    m̂ = rtda.lepski(Xn=Xn, params=θ)
    @test m/10 <= m̂ <= m+n
end

@testset "RobustKDist" begin
    K = ExponentialKernel() ∘ ScaleTransform(2.5)
    w_rkde = rtda.rkde_W(Xn; k=K)
    rkdist = fit(Xn; X=Xn, w=w_rkde, k=K)
    @test typeof(w_rkde) == Vector{Float64}
    @test typeof(rkdist) == Vector{Float64}
    @test sum(w_rkde) ≈ 1.0
    @test length(w_rkde) == N
end

@testset "kPDTM" begin
    q, k, sig = 40, round(Int, N/2), round(Int, N/2)
    Xn .= map(x -> [x...; 0.0], Xn)
    iter_max, nstart = 100, 2
    kPDTM_values, centers, _, _, colors, cost = rtda.kPDTM(Xn, Xn, q, k, sig, iter_max, nstart)
    @test typeof(kPDTM_values) == Vector{Float64}
    @test typeof(centers) == Matrix{Float64}
    @test typeof(colors) == Vector{Float64}
    @test typeof(cost) == Float64
    @test cost >= 0
    @test sum(colors .== -1) > 0
end