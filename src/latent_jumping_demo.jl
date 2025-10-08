using Pkg
Pkg.activate(".")
Pkg.add(url="https://github.com/HedwigNordlinder/ForwardBackward.jl")
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
    # We exp the rates to ensure they are positive, so it essentially training to output log rates
    decoded = l.decode(x)
    tXt .+ decoded[1:end-3,:] .* (1.05f0 .- expand(t, ndims(tXt))), decoded[end-2:end,:]
end
model = FModel(embeddim = 256, layers = 3, spacedim = 1)

η = 0.001
opt_state = Flux.setup(AdamW(η), model)
n_samples = 400
iters = 1000
losses = []

# --- helper: numerically stable log-softmax along dim=1 (rows) ---
logsumexp(A; dims=1) = log.(sum(exp.(A .- maximum(A; dims=dims)), dims=dims)) .+ maximum(A; dims=dims)
logsoftmax(A) = A .- logsumexp(A; dims=1)

# stationary π for your fixed Q (solve Q'π=0, sum π=1)
Astat = [Matrix(Q)'; ones(1,3)]
bstat = vcat(zeros(eltype(Q), 3), one(eltype(Q)))
π_stat = Astat \ bstat               # size (3,)
π_stat = reshape(π_stat, :, 1)       # make it (3,1) for broadcasting

α = 1e-2   # weight for logits regularizer (tune 1e-3—1e-1)


for i in 1:iters
    X0 = SwitchState(ContinuousState(sampleX0(n_samples)), DiscreteState(3,rand(1:3,n_samples)))
    X1 = SwitchState(ContinuousState(sampleX1(n_samples)), DiscreteState(3,fill(2,n_samples)))
    t = rand(T, n_samples)
    Xt = endpoint_conditioned_sample(X0, X1, P, t)
   
    l, g = Flux.withgradient(model) do m
        continuous_prediction, state_prediction = m(t, Xt.main_state) 
        state_augmentation_prediction = (P.possible_jumps' * state_prediction)'
        final_prediction = collect(eachrow(continuous_prediction))[1] .+ state_augmentation_prediction
        floss(CP, final_prediction, Xt.main_state, scalefloss(CP, t))
    end
    Flux.update!(opt_state, model, g[1])
    (i % 100 == 0) && println("Iteration $i, Loss: $l")
    push!(losses, l)
end

plot(losses)
