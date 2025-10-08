using Pkg
Pkg.activate(".")
Pkg.add(url="https://github.com/HedwigNordlinder/ForwardBackward.jl")

using ForwardBackward, Flowfusion, Flux, RandomFeatureMaps, Optimisers, Plots, LinearAlgebra, Distributions
include("plotting.jl")
include("extremesampler.jl")
struct FModel{A}
    layers::A
end
Flux.@layer FModel

function FModel(; embeddim = 128, spacedim = 1, layers = 3)
    embed_time = Chain(RandomFourierFeatures(1 => embeddim, 0.1f0), Dense(embeddim => embeddim, swish))
    embed_state = Chain(RandomFourierFeatures(1 => 4embeddim, 0.1f0), Dense(4*embeddim => embeddim, swish))
    ffs = [Dense(embeddim => embeddim, swish) for _ in 1:layers]
    decode = Dense(embeddim => spacedim)
    layers = (; embed_time, embed_state, ffs, decode)
    FModel(layers)
end

function (f::FModel)(t, Xt)
    l = f.layers
    tXt = tensor(Xt)
    tv = zero(tXt[1:1,:]) .+ expand(t, ndims(tXt))
    x = l.embed_time(tv) .+ l.embed_state(tXt)
    for ff in l.ffs
        x = x .+ ff(x)
    end
    #l.decode(x) # GPT claims this will work, I highly doubt it
    tXt .+ l.decode(x) .* (1.05f0 .- expand(t, ndims(tXt)))
end

model = FModel(embeddim = 256, layers = 3, spacedim = 1)

T = Float32

X0_distribution = Uniform()
X1_distribution = MixtureModel([Uniform(1,2),Uniform(-1,0)],[1/2,1/2])

sampleX0(n_samples) = Float32.(rand(X0_distribution, 1, n_samples)) 
sampleX1(n_samples) = Float32.(rand(X1_distribution,1,n_samples))

n_samples = 1600
CP = BrownianMotion(0.1f0)
Q_function(X::ContinuousState) = T.(min((1f0 + norm(X.state)),3f0) .* [-1/4 1/4; 1/4 -1/4])
P = XDependentSwitchBridgeProcess(CP, Q_function)

η = 0.001
opt_state = Flux.setup(AdamW(η), model)

iters = 1000
losses = []

# Pre-allocate arrays outside the loop for better performance
target_states = similar(sampleX0(n_samples))
target_array = ContinuousState(target_states)

for i in 1:iters
    X0 = SwitchState(ContinuousState(T.(rand(X0_distribution, 1, n_samples))), DiscreteState(2,rand(1:2,n_samples)))
    X1 = SwitchState(ContinuousState(sampleX1(n_samples)), DiscreteState(2,fill(1,n_samples)))
    t = rand(T, n_samples)
    Xt = endpoint_conditioned_sample(X0, X1, P, t)
    
    # Optimized preprocessing with views and in-place operations
    switch_mask = @view(Xt.switching_state.state[:,1]) .== 1
    @views begin
        target_states[:, switch_mask] .= X1.main_state.state[:, switch_mask]
        target_states[:, .!switch_mask] .= X0.main_state.state[:, .!switch_mask]
    end

    l, g = Flux.withgradient(model) do m
        floss(CP, m(t,Xt.main_state), target_array, scalefloss(CP, t))
    end
    Flux.update!(opt_state, model, g[1])
    (i % 100 == 0) && println("Iteration $i, Loss: $l")
    push!(losses, l)
end

n_inference_samples = 5000
X0 = ContinuousState(T.(rand(X0_distribution, 1, n_inference_samples)))
paths = Tracker()
println("Starting sampling...")
samples = gen(CP, X0, model, 0f0:0.005f0:1f0, tracker = paths)
tvec = stack_tracker(paths, :t)
xttraj = stack_tracker(paths, :xt)

plot(losses)
println("Plotting endpoints...")
plot_vector_endpoints(X0.state, samples.state)
println("Plotting trajectories...")

println("Plotting histogram...")
plot_histogram_vs_true_density(samples.state, X1_distribution)

# Generate GIF of histogram evolution
println("Generating histogram evolution GIF...")
plot_histogram_evolution_gif(tvec, xttraj, X1_distribution, 
                            filename="histogram_evolution.gif",
                            fps=10,
                            every_n_frames=3,
                            title_prefix="Time: ")


simulations = 10000
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

X0 = SwitchState(ContinuousState(sampleX0(simulations)), DiscreteState(2,rand(1:2,simulations)))
X1 = SwitchState(ContinuousState(sampleX1(simulations)), DiscreteState(2,fill(1,simulations)))

#endpoint_conditioned_sample(X0, X1, P, simulation_times; tracker = conditional_path_tracker)

# Test the new plot_trajectories function with conditional data
#println("Plotting conditional trajectories...")
#plot_trajectories_conditional(conditional_path_times, conditional_path_states, n_plot=10000, alpha=0.0075)

plot_trajectories_standard(tvec, xttraj; x1_distribution = X1_distribution)