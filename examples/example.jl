using RobustTDA
using Ripserer, Pipe, Plots

begin
    Random.seed!(2022)
    win = (-1, 1, -1, 1)
    signal = 2 .* randCircle(500, sigma = 0.05)
    noise = randMClust(100, window = win, λ1 = 2, λ2 = 10, r = 0.05)
    X = [signal; noise]
    points = [signal; noise]
    scatter(points, ratio=1, markeralpha=0.5, label=nothing)
end

begin
    Xn = [[x...] for x in points]
end

begin
    f_dist = RobustTDA.dist(Xn)
    f_momdist = momdist(Xn, 201)
    f_dtm = dtm(Xn, 1/6)
    f = f_momdist
end

begin
    plt = f -> plot(xseq, yseq, (x,y) -> fit([[x,y]],f), st=:surface, camera=(10,60))
    @pipe [f_dist; f_momdist; f_dtm] .|> plt |> plot(_..., layout=(1,3), size=(900,200))
end



begin
    plot(xseq, yseq, (x,y) -> fit([[x,y]],f), st=:surface, camera=(10,60), fillalpha=1)
end

begin
    xseq = -3:0.05:3
    yseq = -3:0.05:3
    F_grid = [RobustTDA.fit([[x, y]], f) for x in xseq, y in yseq]
    D_sublevel = F_grid |> Cubical |> ripserer
    plot(D_sublevel)
end


begin
    # f-weighted iltration value 't' and power 'p'
    t = 1.2
    p = 1
    weights_f = fit(Xn, f)
    Bt = Balls(Xn, rfx.(t, weights_f, p))
    plot(Bt, c=:orange, ratio=1, fillalpha=1, linealpha=0, markeralpha=0)
end


