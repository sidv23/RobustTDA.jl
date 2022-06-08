function plot_balls(B; X = nothing, p = nothing, par::plot_params)
    if isnothing(X) & isnothing(p)
        p = plot(B,
        c = par.col, fillalpha = par.alpha, linewidth = par.lwd,
        linealpha = par.lalpha, linecolor = par.col, ratio = 1)
        
    elseif isnothing(X)
        p = plot(B,
        c = par.col, fillalpha = par.alpha, linewidth = par.lwd,
        linealpha = par.lalpha, linecolor = par.col, ratio = 1)
        
    elseif isnothing(p)
        p = scatter(X, ratio = 1, label = nothing)
        p = plot(p, B,
        c = par.col, fillalpha = par.alpha, linewidth = par.lwd,
        linealpha = par.lalpha, linecolor = par.col, ratio = 1)
        
    else
        p = scatter(p, X, ratio = 1, label = nothing)
        p = plot(B,
        c = par.col, fillalpha = par.alpha, linewidth = par.lwd,
        linealpha = par.lalpha, linecolor = par.col, ratio = 1)
    end
    
    return p
end



function filtration_plot(t;
    Xn::Vector{<:Vector{<:Real}},
    w::Vector{<:Real} = nothing,
    p = 1,
    par::plot_params,
    clim=nothing
    )
    
    if isnothing(w)
        w = repeat([0], length(Xn))
    end
    
    if isnothing(clim)
        clim = extrema(w)
    end
    
    plt = scatter(Tuple.(Xn), marker_z = w, label = nothing, ratio = 1, clim=clim)
    plt = plot_balls(Balls(Xn, rfx.(t, w, p)), p = plt, par = par)
    plt = scatter(plt, Tuple.(Xn), marker_z = w, label = nothing, ratio = 1, clim=clim)
    
    return plt
end


function surfacePlot(xseq::Any, yseq = nothing; f::Function, args...)
    if yseq |> isnothing
        yseq = xseq
    end
    plt = plot(xseq, yseq, f, st = :surface; args...)
    display(plt)
    return plt
end