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


# function all_filtration_plots(t; Xn, W, p, lim = (-4.0, 4.0), plot_par::plot_params = nothing)

#     if isnothing(plot_par)
#         plot_par = plot_params(alpha = 0.3)
#     end

#     # Vanilla
#     plt1 = filtration_plot(t; Xn = Xn, w = repeat([0], length(Xn)), p = p, par = plot_par)
#     plt1 = plot(plt1, title = "Vanilla", xlim = lim, ylim = lim)

#     # KDIST
#     plt2 = filtration_plot(t; Xn = Xn, w = W["kdist"], p = p, par = plot_par) #, clim = clims)
#     plt2 = plot(plt2, title = "KDist", xlim = lim, ylim = lim)

#     # MOM KDIST
#     plt3 = filtration_plot(t; Xn = Xn, w = W["momkdist"], p = p, par = plot_par)
#     plt3 = plot(plt3, title = "MoM KDist", xlim = lim, ylim = lim)

#     # DTM
#     plt4 = filtration_plot(t; Xn = Xn, w = W["dtm"], p = p, par = plot_par)
#     plt4 = plot(plt4, title = "DTM", xlim = lim, ylim = lim)

#     # MOMDTM
#     plt5 = filtration_plot(t; Xn = Xn, w = W["momdtm"], p = p, par = plot_par) #, clim = clims)
#     plt5 = plot(plt5, title = "MOM DTM", xlim = lim, ylim = lim)

#     # MOMDIST
#     plt6 = filtration_plot(t; Xn = Xn, w = W["momdist"], p = p, par = plot_par)
#     plt6 = plot(plt6, title = "MoM Dist", xlim = lim, ylim = lim)

#     layout = @layout (2, 3)
#     plt = plot(plt1, plt2, plt3, plt4, plt5, plt6, layout = layout)

#     return plt
# end




# function all_diagram_plots(D, lim = (0, 4.0); dims=nothing)

#     if isnothing(dims)
#         dims = length(D["vanilla"])
#     end

#     # Vanilla
#     plt1 = plot(D["vanilla"][dims], title = "Vanilla", xlim = lim, ylim = lim)

#     # KDIST
#     plt2 = plot(D["kdist"][dims], title = "kdist", xlim = lim, ylim = lim)

#     # MOM KDIST
#     plt3 = plot(D["momkdist"][dims], title = "momkdist", xlim = lim, ylim = lim)

#     # DTM
#     plt4 = plot(D["dtm"][dims], title = "dtm", xlim = lim, ylim = lim)

#     # MOMDTM
#     plt5 = plot(D["momdtm"][dims], title = "momdtm", xlim = lim, ylim = lim)

#     # MOMDIST
#     plt6 = plot(D["momdist"][dims], title = "momdist", xlim = lim, ylim = lim)

#     layout = @layout (2, 3)
#     plt = plot(plt1, plt2, plt3, plt4, plt5, plt6, layout = layout)

#     return plt
# end




function surfacePlot(xseq::Any, yseq = nothing; f::Function, args...)
    if yseq |> isnothing
        yseq = xseq
    end
    plt = plot(xseq, yseq, f, st = :surface; args...)
    display(plt)
    return plt
end