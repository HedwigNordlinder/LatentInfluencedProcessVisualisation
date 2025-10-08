using Pkg
Pkg.activate(".")
Pkg.add(url="https://github.com/HedwigNordlinder/ForwardBackward.jl")
Pkg.add(url="https://github.com/HedwigNordlinder/Flowfusion.jl")
using ForwardBackward, Distributions, Flowfusion, Flux, RandomFeatureMaps, Optimisers, Plots, LinearAlgebra
include("plotting.jl")

t = 0f0:0.005f0:1f0
X0_distribution = Normal(0,5)
X1_distribution = MixtureModel([Normal(5,0.5),Normal(-5,10)],[1/2,1/2])

sampleX0(n) = Float32.(rand(X0_distribution, 1, n))
sampleX1(n) = Float32.(rand(X1_distribution, 1, n))

CP = BrownianMotion(1f0)
λ = 10.0f0
μ = 10.0f0
T = Float32
Q = T.([-λ λ 0; μ (-μ-λ) λ; 0 μ -μ])
DP = GeneralDiscrete(Q)

possible_jumping_array = T.([-1, 0, 1])

P = LatentJumpingProcess(CP, DP, possible_jumping_array)

