# RobustTDA.jl

module RobustTDA

# Dpendencies
using Random
using Distributions
using Plots
using Parameters
using Distances
using LinearAlgebra
using Pipe
using Statistics
using RCall
using Ripserer
using LazySets
using PersistenceDiagrams
using NearestNeighbors
using PersistenceDiagramsBase
using Markdown
using KernelFunctions
using MLDataUtils
using Roots
using LambertW

import Base: log

export randCircle,
    randLemniscate,
    randUnif,
    randMClust,
    plot_params,
    wRips_params,
    DistanceLikeFunction,
    _Matrix_to_ArrayOfTuples,
    _ArrayOfTuples_to_Matrix,
    _ArrayOfVectors_to_ArrayOfTuples,
    _ArrayOfTuples_to_ArrayOfVectors,
    surfacePlot,
    partition,
    plot_balls,
    filtration_plot,
    Balls,
    rfx,
    convert_radius,
    dH,
    MMD,
    KDist,
    momKDist,
    dtm,
    momdtm,
    momdist,
    bandwidth_select,
    fit,
    test,
    surfacePlot,
    rkde,
    log,
    lepski


include("structures.jl")
include("functions/shapes.jl")
include("functions/helperFunctions.jl")
include("functions/plotFunctions.jl")
include("functions/distanceFunctions.jl")
include("functions/distanceLikeFunctions.jl")
include("functions/tdaFunctions.jl")
include("functions/transformations.jl")
include("functions/rkde.jl")
include("functions/autotune.jl")

end
