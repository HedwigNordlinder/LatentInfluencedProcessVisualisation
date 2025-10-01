using Plots, Random

"""
    plot_vector_endpoints(vec1, vec2; kwargs...)

Plots two vectors/matrices in 2D space with time on X-axis and state on Y-axis:
- `vec1`: Points plotted at the far left (time = 0)
- `vec2`: Points plotted at the far right (time = 1)

# Arguments
- `vec1`: First vector or matrix (1×n) to plot at time 0
- `vec2`: Second vector or matrix (1×n) to plot at time 1
- `markersize`: Size of the markers (default: 6)
- `color1`: Color for first vector points (default: :blue)
- `color2`: Color for second vector points (default: :red)
- `label1`: Label for first vector (default: "t=0")
- `label2`: Label for second vector (default: "t=1")

# Example
```julia
vec1 = randn(1, 100) .- 1
vec2 = randn(1, 100) .+ 1
plot_vector_endpoints(vec1, vec2)
```
"""
function plot_vector_endpoints(vec1, vec2; 
                                markersize=6, 
                                color1=:blue, 
                                color2=:red,
                                label1="t=0",
                                label2="t=1")
    # Convert to vectors if matrices
    state1 = vec(vec1)
    state2 = vec(vec2)
    
    n1 = length(state1)
    n2 = length(state2)
    
    # First vector: plot at time t=0
    t1 = zeros(n1)
    
    # Second vector: plot at time t=1
    t2 = ones(n2)
    
    # Create 2D plot
    p = scatter(t1, state1, 
                markersize=markersize, 
                color=color1, 
                label=label1,
                xlabel="Time", 
                ylabel="State",
                legend=:best)
    
    scatter!(p, t2, state2, 
             markersize=markersize, 
             color=color2, 
             label=label2)
    
    return p
end

"""
    plot_trajectories(tvec, xttraj; kwargs...)

Plots trajectories of points over time.

# Arguments
- `tvec`: Time vector of length T
- `xttraj`: Array of shape (state_dim, n_trajectories, n_timesteps) containing trajectories
- `n_plot`: Number of trajectories to plot (default: all, or 500 if more than 500)
- `alpha`: Transparency of trajectory lines (default: 0.3)
- `linewidth`: Width of trajectory lines (default: 1)
- `color`: Color of trajectories (default: :blue)
- `legend`: Show legend (default: false)

# Example
```julia
tvec = 0.0f0:0.005f0:1.0f0
xttraj = randn(Float32, 1, 5000, 200)
plot_trajectories(tvec, xttraj)
```
"""
function plot_trajectories(tvec, xttraj; 
                          n_plot=50,
                          alpha=0.3,
                          linewidth=1,
                          color=:blue,
                          legend=false)
    state_dim, n_trajectories, n_timesteps = size(xttraj)
    
    # For 1D state space
    if state_dim != 1
        error("Currently only supports 1D state space (state_dim = 1)")
    end
    
    # Determine how many trajectories to plot
    if n_plot === nothing
        n_plot = n_trajectories > 500 ? 500 : n_trajectories
    end
    n_plot = min(n_plot, n_trajectories)
    
    # Select random subset if needed
    if n_plot < n_trajectories
        indices = sort(shuffle(1:n_trajectories)[1:n_plot])
    else
        indices = 1:n_trajectories
    end
    
    # Create plot
    p = plot(xlabel="Time", 
             ylabel="State",
             legend=legend)
    
    # Plot each trajectory
    for idx in indices
        trajectory = xttraj[1, idx, :]
        plot!(p, tvec, trajectory, 
              alpha=alpha, 
              linewidth=linewidth,
              color=color,
              label="")
    end
    
    return p
end

export plot_vector_endpoints, plot_trajectories

