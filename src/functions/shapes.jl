using Random, Distributions, RCall


############################################################
# LEMNISCATE
function randLemniscate(n; sigma = 0)

    signal = R"tdaunif::sample_lemniscate_gerono($n)"
    noise = R"matrix(rnorm(2 * $n, 0, $sigma), ncol=2)"

    X = tuple.(eachcol(rcopy(signal + noise))...)
    # X = [eachrow( rcopy( signal + noise ) )...]

    return X
end

############################################################
# CIRCLE
function randCircle(n; sigma = 0)

    signal = R"TDA::circleUnif($n)"
    noise = R"matrix(rnorm(2 * $n, 0, $sigma), ncol=2)"

    X = tuple.(eachcol(rcopy(signal + noise))...)
    # X = [eachrow( rcopy( signal + noise ) )...]

    return X
end


############################################################
# UNIFORM DISTRIBUTION
function randUnif(n; a = 0, b = 1, d = 2)
    return [tuple(rand(Uniform(a, b), d)...) for _ ∈ 1:1:n]
    # return [ rand(Uniform(a, b), d) for _ ∈ 1:1:n ]
end


############################################################
# MATÉRN CLUSTER
# 
# The following code is taken as is from H. Paul Keeler's implementation of the
# Matérn cluster process in Julia. All credit goes to Paul.
# See: 
#   https://hpaulkeeler.com/simulating-a-matern-cluster-point-process/
#   https://github.com/hpaulkeeler/posts/blob/master/TestingJulia/MaternClusterRectangle.jl
# 


function randMClust(n; window = (-1, 1, -1, 1), λ1 = 5, λ2 = 5, r = 0.1)
    # Simulation window parameters

    xMin, xMax, yMin, yMax = window

    lambdaParent = λ1
    lambdaDaughter = λ2
    radiusCluster = r

    # xMin = -.5;
    # xMax = .5;
    # yMin = -.5;
    # yMax = .5;

    # # Parameters for the parent and daughter point processes
    # lambdaParent = 10;# density of parent Poisson point process
    # lambdaDaughter = 10;# mean number of points in each cluster
    # radiusCluster = 0.01;# radius of cluster disk (for daughter points)

    # Extended simulation windows parameters
    rExt = radiusCluster # extension parameter -- use cluster radius
    xMinExt = xMin - rExt
    xMaxExt = xMax + rExt
    yMinExt = yMin - rExt
    yMaxExt = yMax + rExt
    # rectangle dimensions
    xDeltaExt = xMaxExt - xMinExt
    yDeltaExt = yMaxExt - yMinExt
    areaTotalExt = xDeltaExt * yDeltaExt # area of extended rectangle

    # Simulate Poisson point process
    numbPointsParent = rand(Poisson(areaTotalExt * lambdaParent)) # Poisson number of points

    # x and y coordinates of Poisson points for the parent
    xxParent = xMinExt .+ xDeltaExt * rand(numbPointsParent)
    yyParent = yMinExt .+ yDeltaExt * rand(numbPointsParent)

    # Simulate Poisson point process for the daughters (ie final poiint process)
    numbPointsDaughter = rand(Poisson(lambdaDaughter), numbPointsParent)
    numbPoints = sum(numbPointsDaughter) # total number of points

    # Generate the (relative) locations in polar coordinates by
    # simulating independent variables.
    theta = 2 * pi * rand(numbPoints) # angular coordinates
    rho = radiusCluster * sqrt.(rand(numbPoints)) # radial coordinates

    # Convert polar to Cartesian coordinates
    xx0 = rho .* cos.(theta)
    yy0 = rho .* sin.(theta)

    # replicate parent points (ie centres of disks/clusters)
    xx = vcat(fill.(xxParent, numbPointsDaughter)...)
    yy = vcat(fill.(yyParent, numbPointsDaughter)...)

    # Shift centre of disk to (xx0,yy0)
    xx = xx .+ xx0
    yy = yy .+ yy0

    # thin points if outside the simulation window
    booleInside = ((xx .>= xMin) .& (xx .<= xMax) .& (yy .>= yMin) .& (yy .<= yMax))
    # retain points inside simulation window
    xx = xx[booleInside]
    yy = yy[booleInside]

    X = tuple.(eachcol([xx yy])...)

    return rand(X, n)
end
