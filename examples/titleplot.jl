using RobustTDA
using Plots
using Pipe
using Random

begin
    Random.seed!(2022)
    
    n = 500
    m = 100

    p1 = (1, 0)
    p2 = (-1, 0)
    p3 = (0, 2 * sin( 2*π/3 ))
    
    X1 = @pipe randCircle(n, sigma=0.04) .|> ( _.*0.5 .+ p1 )
    X2 = @pipe randCircle(n, sigma=0.04) .|> ( _.*0.5 .+ p2 )
    X3 = @pipe randCircle(n, sigma=0.04) .|> ( _.*0.5 .+ p3 )
    # noise = randMatern(50, a=-2, b=2)
    noise = randMClust(50, window=(-2,2,-2,2))
    X_signal = [X1[1:m];X2[1:m];X3[1:m]]
    X = [X1; X2; X3; noise]
    Xn = _ArrayOfTuples_to_ArrayOfVectors(X)
end


begin
    w = fit(Xn, momdist(Xn, 102))
    # w = fit(Xn, dtm(Xn, 0.05))
    scatter(noise, ratio=1, axis=false, grid=false, label=false, ticks=false, markersize=3, c=:black)
    scatter!(X_signal, ratio=1, axis=false, grid=false, label=false, ticks=false, markersize=3, c=:black)
    # scatter!(X, marker_z=w, ratio=1, axis=false, grid=false, label=false, ticks=false, markersize=3, c=:black)
end

begin
    w1 = w[1:m]
    w2 = w[(n+1):(n+m)]
    w3 = w[(2*n+1):(2*n+m)]
    w4 = w[(end-50+1):end]
end

begin
    t = 0.4
    p = 1
    B1 = Balls(X1, rfx.(t, w1, p))
    B2 = Balls(X2, rfx.(t, w2, p))
    B3 = Balls(X3, rfx.(t, w3, p))
    B4 = Balls(noise, rfx.(t, w4, p))
end

begin
    plt = plot(0,0,markeralpha=0,ratio=1, axis=false, grid=false, label=false, ticks=false)
    plt = @pipe B4 |> plot(plt, _, c=:black, linealpha=0, ratio=1, alpha=0.2)
    plt = @pipe B1 |> plot(plt, _, c=:purple, linealpha=0, alpha=1)
    plt = @pipe B2 |> plot(plt, _, c=:firebrick1, linealpha=0, alpha=1)
    plt = @pipe B3 |> plot(plt, _, c=:green, linealpha=0, alpha=1)
    scatter!(plt,noise, ratio=1, axis=false, grid=false, label=false, ticks=false, markersize=2, c=:gray)
    scatter!(X_signal, ratio=1, axis=false, grid=false, label=false, ticks=false, markersize=2, c=:gray)
end

plot(plt, background_color = :transparent, foreground_color=:black)

savefig("./examples/logo.svg")