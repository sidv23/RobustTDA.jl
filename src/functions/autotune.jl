"""
    lepski(Xn, params::lepski_params)

Implements Lepski's adaptive selection procedure to determine an optimal parameter `m` for estimation.

# Arguments
- `Xn`: A dataset represented as an array.
- `params::lepski_params`: A structure containing the following parameters:
    - `a`: A constant used in the function `h`.
    - `b`: A constant exponent in the function `h`.
    - `mmin`: The minimum value of `m`.
    - `mmax`: The maximum value of `m`.
    - `pi`: A constant used for generating values of `m`.
    - `δ`: A probability threshold.

# Returns
- The optimal value of `m` selected by Lepski's principle.

# Description
This function iteratively computes persistence diagrams for different values of `m` and selects the best one using the Bottleneck distance criterion. It constructs a sequence of values for `m` based on geometric scaling and evaluates differences between persistence diagrams to find the smallest `m` where the difference falls below a threshold. The function utilizes progress tracking and garbage collection to optimize performance.

# Example Usage
```julia
    θ = rtda.lepski_params(
        a=0.2,
        b=1,
        mmin=50,
        mmax=200,
        pi=1.1,
        δ=0.01
    )
    m̂ = rtda.lepski(Xn=Xn, params=θ)
```
"""
function lepski(; Xn, params::lepski_params)

    p = 1
    @unpack a, b, mmin, mmax, pi, δ = params
    n = length(Xn)
    δmin = exp(-8 * (1 + b) * (2 * mmin + 1))

    h = (n, m, δ) ->
        2 * ((2 * m + 1) / (a * n) * lambertw((n * δmin) / (2 * m + 1)))^(1 / b) +
        ((1) / (a * (n - m)) * lambertw((n - m) / ((δ - δmin)^4)))^(1 / b)

    M = [round(Int, mmin * pi^j) for j in 1:1:floor(Int, log(pi, mmax / mmin))] |> unique
    Q = [2 * m + 1 for m in M]
    J = length(M)
    D = @showprogress "Computing Dgms" [@pipe q .|> momdist(Xn, _) .|> fit(Xn, _) .|> wrips(Xn, w=_, p=1) for q in Q]

    jhat = J

    prog = Progress(convert(Int, J * (J - 1) / 2))
    generate_showvalues(j) = () -> [(:m, M[j])]

    for j in 1:(J-1)
        flag = false
        # Dj = @pipe Q[j] |> momdist(Xn, _) |> fit(Xn, _) |> wrips(Xn, w = _, p = 1)
        for i in (j+1):J
            # Di = @pipe Q[i] |>    momdist(Xn, _) |> fit(Xn, _) |> wrips(Xn, w = _, p = 1)
            # flag = Bottleneck()(Di, Dj) > 2 * h(n, M[i])


            flag = Bottleneck()(D[i][1], D[j][1]) ≤ 2 * h(n, M[i], δ)
            # flag = Bottleneck()(D[i], D[j]) ≤ 2 * h(n, M[i], δ)
            next!(prog; showvalues=generate_showvalues(j))
        end

        if flag
            jhat = j
            break
        end
    end

    GC.gc()

    return M[jhat]
end