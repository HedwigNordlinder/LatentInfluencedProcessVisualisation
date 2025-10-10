using Pkg
Pkg.activate(".")
#Pkg.add(url="https://github.com/HedwigNordlinder/ForwardBackward.jl")
#Pkg.add(url="https://github.com/HedwigNordlinder/Flowfusion.jl")
using ForwardBackward, Distributions, Flowfusion, Flux, RandomFeatureMaps, Optimisers, Plots, LinearAlgebra
include("plotting.jl")

t = 0f0:0.005f0:1f0
X0_distribution = Normal(0,5)
X1_distribution = MixtureModel([Normal(5,0.5),Normal(-5,0.5)],[1/2,1/2])

sampleX0(n) = Float32.(rand(X0_distribution, 1, n))
sampleX1(n) = Float32.(rand(X1_distribution, 1, n))

CP = BrownianMotion(5f0)
λ = 3.0f0
μ = 3.0f0
T = Float32
Q = T.([-λ λ 0; μ (-μ-λ) λ; 0 μ -μ])
#DP = GeneralDiscrete(Q)
DP = UniformDiscrete(5f0)
possible_jumping_array = T.([-5, 0, 1])

P = LatentJumpingProcess(CP, DP, possible_jumping_array)

struct FModel{A}
    layers::A
end
Flux.@layer FModel

function FModel(; embeddim = 128, spacedim = 1, layers = 3)
    embed_time = Chain(RandomFourierFeatures(1 => embeddim, 1f0), Dense(embeddim => embeddim, swish))
    embed_state = Chain(RandomFourierFeatures(1 => 4embeddim, 1f0), Dense(4*embeddim => embeddim, swish))
    ffs = [Dense(embeddim => embeddim, swish) for _ in 1:layers]
    # The "plus one" here is for the latent rates. They ought to be positive,
    # so we maybe should enforce that somehow. 
    decode = Dense(embeddim => spacedim + 3) 
    layers = (; embed_time, embed_state, ffs, decode)
    FModel(layers)
end

softmax(x) = exp.(x .- maximum(x)) ./ sum(exp.(x .- maximum(x)))
function (f::FModel)(t, Xt)
    l = f.layers
    tXt = tensor(Xt)
    tv = zero(tXt[1:1,:]) .+ expand(t, ndims(tXt))
    x = l.embed_time(tv) .+ l.embed_state(tXt)
    for ff in l.ffs
        x = x .+ ff(x)
    end
    len = size(tXt, 2)
    # We exp the rates to ensure they are positive, so it essentially training to output log rates
    decoded = l.decode(x)
    tXt .+ decoded[1:end-3,:] .* (1.05f0 .- expand(t, ndims(tXt))), reshape(decoded[end-2:end,:],:,1,len)
end
model = FModel(embeddim = 256, layers = 3, spacedim = 1)

η = 0.001
opt_state = Flux.setup(AdamW(η), model)
n_samples = 400
iters = 1000
losses = []

wrapped_DP = DoobMatchingFlow(DP)
for i in 1:iters
    cont_X0_state = sampleX0(n_samples)
    cont_X1_state = sampleX1(n_samples)
    X0 = LatentJumpingState(ContinuousState(copy(cont_X0_state)), DiscreteState(3,rand(1:3,n_samples)), ContinuousState(copy(cont_X0_state)))
    X1 = LatentJumpingState(ContinuousState(copy(cont_X1_state)), DiscreteState(3,fill(2,n_samples)), ContinuousState(copy(cont_X1_state)))
    t = rand(T, n_samples)
    Xt = endpoint_conditioned_sample(X0, X1, P, t)
    
    G = Guide(wrapped_DP, t, Xt.switching_state, onehot(X1.switching_state))
    l, g = Flux.withgradient(model) do m
        continuous_prediction, state_prediction = m(t, Xt.combined_state) 
        #state_augmentation_prediction = (P.possible_jumps' * state_prediction)'
        #final_prediction = collect(eachrow(continuous_prediction))[1] .+ state_augmentation_prediction
        floss(CP, continuous_prediction, X1.continuous_state, scalefloss(CP, t)) + floss(wrapped_DP, onehot(Xt.switching_state),state_prediction,G, scalefloss(DP, t))
    end
    Flux.update!(opt_state, model, g[1])
    (i % 100 == 0) && println("Iteration $i, Loss: $l")
    push!(losses, l)
end

plot(losses)

X0_samples = sampleX0(1000)
X0 = LatentJumpingState(ContinuousState(copy(X0_samples)), DiscreteState(3,fill(2,1000)), ContinuousState(copy(X0_samples)))

function gen_samples(P::LatentJumpingProcess, X0::LatentJumpingState, model, t; ϵ = 1e-2)
    n = length(X0.combined_state.state)
    tvec = collect(t)
    xt = copy(X0)
    samples = [xt]
    times = tvec
    for i in 1:(length(tvec) - 1)
        current_time = fill(tvec[i],n)
        next_time = fill(tvec[i+1],n)
        model_prediction =Flowfusion.resolveprediction(model(current_time, xt.combined_state.state), X0)
        xt = endpoint_conditioned_sample(xt, model_prediction, P, current_time, next_time)
        push!(samples, xt)
    end
    return samples, times
end

samples, times = gen_samples(P, X0, model, 0f0:0.005f0:1f0)
plot_trajectories_standard(times, [sample.combined_state.state for sample in samples])

plot_histogram_evolution_gif(times, [sample.combined_state.state for sample in samples], X1_distribution;
filename="evolution.gif", fps=10, every_n_frames=3)