using Pkg
Pkg.activate(".")
Pkg.add(url="https://github.com/HedwigNordlinder/ForwardBackward.jl")

using ForwardBackward, Flowfusion, Flux, RandomFeatureMaps, Optimisers, Plots, LinearAlgebra
include("plotting.jl")

struct FModel{A}
    layers::A
end
Flux.@layer FModel

function FModel(; embeddim = 128, spacedim = 1, layers = 3)
    embed_time = Chain(RandomFourierFeatures(1 => embeddim, 1f0), Dense(embeddim => embeddim, swish))
    embed_state = Chain(RandomFourierFeatures(1 => embeddim, 1f0), Dense(embeddim => embeddim, swish))
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
    tXt .+ l.decode(x) .* (1.05f0 .- expand(t, ndims(tXt)))
end

model = FModel(embeddim = 256, layers = 3, spacedim = 1)

T = Float32
sampleX0(n_samples) = randn(T, 1, n_samples) .- 1
sampleX1(n_samples) = randn(T, 1, n_samples) .+ 1

n_samples = 400
CP = Deterministic()
Q_function(X::ContinuousState) = T.((1+norm(X.state)) .* [-5 5; 5 -5])
P = XDependentSwitchBridgeProcess(CP, Q_function)

η = 0.001
opt_state = Flux.setup(AdamW(η), model)

iters = 1000
losses = []

for i in 1:iters
    X0 = SwitchState(ContinuousState(sampleX0(n_samples)), DiscreteState(2,rand(1:2, n_samples)))
    X1 = SwitchState(ContinuousState(sampleX1(n_samples)), DiscreteState(2,fill(1,n_samples)))
    t = rand(T, n_samples)
    Xt = endpoint_conditioned_sample(X0, X1, P, t)
    switch_mask = Xt.switching_state.state[:,1] .== 1
    target_states = similar(X0.main_state.state)
    target_states[:, switch_mask] = X1.main_state.state[:, switch_mask]
    target_states[:, .!switch_mask] = X0.main_state.state[:, .!switch_mask]
    target_array = ContinuousState(target_states)

    l, g = Flux.withgradient(model) do m
        floss(CP, m(t,Xt.main_state), target_array, scalefloss(CP, t))
    end
    Flux.update!(opt_state, model, g[1])
    (i % 100 == 0) && println("Iteration $i, Loss: $l")
    push!(losses, l)
end

n_inference_samples = 5000
X0 = ContinuousState(sampleX0(n_inference_samples))
paths = Tracker()
samples = gen(Deterministic(), X0, model, 0f0:0.005f0:1f0, tracker = paths)
tvec = stack_tracker(paths, :t)
xttraj = stack_tracker(paths, :xt)

plot(losses)
plot_vector_endpoints(X0.state, samples.state)
plot_trajectories(tvec, xttraj)