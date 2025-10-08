using Pkg
Pkg.activate(".")
Pkg.add(url="https://github.com/HedwigNordlinder/ForwardBackward.jl")

using ForwardBackward, Distributions
include("plotting.jl")

t = 0f0:0.005f0:1f0
X0_distribution = Normal(0,5)
X1_distribution = MixtureModel([Normal(5,0.5),Normal(-5,1)],[1/2,1/2])

sampleX0(n) = Float32.(rand(X0_distribution, 1, n))
sampleX1(n) = Float32.(rand(X1_distribution, 1, n))

CP = BrownianMotion(1f0)
λ = 100f0
μ = 100f0
T = Float32
Q = T.([-λ λ 0; μ (-μ-λ) λ; 0 μ -μ])
DP = GeneralDiscrete(Q)

possible_jumping_array = T.([-1, 0, 1])

P = LatentJumpingProcess(CP, DP, possible_jumping_array)

simulations = 5000
simulation_times = Float32.(fill(1.0f0, simulations))
conditional_path_times = Vector{Vector{Float32}}()
conditional_path_states = Vector{Vector{SwitchState}}()
# We need to pre-allocate these arrays to have as many vectors as we have simulations
for i in 1:length(simulation_times)
    push!(conditional_path_times, [])
    push!(conditional_path_states, [])
end
conditional_path_tracker = function (t,Xt, i)
    push!(conditional_path_times[i], t)
    push!(conditional_path_states[i], Xt)
end

X0 = SwitchState(ContinuousState(sampleX0(simulations)), DiscreteState(3,fill(2,simulations)))
X1 = SwitchState(ContinuousState(sampleX1(simulations)), DiscreteState(3,fill(2,simulations)))
endpoint_conditioned_sample(X0, X1, P, simulation_times; tracker = conditional_path_tracker, ϵ=1e-2)
plot_trajectories_conditional(conditional_path_times, conditional_path_states, n_plot=5000, alpha=0.01)