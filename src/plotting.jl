using Plots, Random, Distributions, Statistics

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
    plot_histogram_vs_true_density(points, distribution; kwargs...)

Plots the histogram of points over the true density of the distribution.

# Arguments
- `points`: Vector of points to create histogram from
- `distribution`: Distribution object from Distributions.jl (e.g., Normal, MixtureModel)
- `n_points`: Number of evaluation points for true density curve (default: 200)
- `xlim`: X-axis limits (default: automatic based on data)
- `hist_color`: Color for histogram (default: :blue)
- `true_color`: Color for true density curve (default: :red)
- `hist_label`: Label for histogram (default: "Histogram")
- `true_label`: Label for true density curve (default: "True Density")
- `linewidth`: Line width for true density curve (default: 2)
- `alpha`: Transparency for histogram (default: 0.7)
- `bins`: Number of histogram bins (default: 30)
- `normalize`: Histogram normalization (:pdf, :density, :probability) (default: :pdf)

# Example
```julia
using Distributions
points = randn(1000)
true_dist = Normal(0, 1)
plot_histogram_vs_true_density(points, true_dist)
```
"""
function plot_histogram_vs_true_density(points, distribution; 
                                       n_points=200,
                                       xlim=nothing,
                                       hist_color=:blue,
                                       true_color=:red,
                                       hist_label="Histogram",
                                       true_label="True Density",
                                       linewidth=2,
                                       alpha=0.7,
                                       bins=100,
                                       normalize=:pdf)
    
    # Convert points to vector if needed
    points_vec = vec(points)
    
    # Determine x-axis limits
    if xlim === nothing
        data_min, data_max = extrema(points_vec)
        margin = 0.1 * (data_max - data_min)
        x_min = data_min - margin
        x_max = data_max + margin
    else
        x_min, x_max = xlim
    end
    
    # Create evaluation points for true density
    x_eval = range(x_min, x_max, length=n_points)
    
    # Compute true density
    true_density = pdf.(distribution, x_eval)
    
    # Create plot
    p = plot(xlabel="Value", 
             ylabel="Density",
             legend=:topright,
             linewidth=linewidth)
    
    # Plot histogram
    histogram!(p, points_vec, 
              alpha=alpha, 
              color=hist_color, 
              label=hist_label,
              normalize=normalize,
              bins=bins)
    
    # Plot true density
    plot!(p, x_eval, true_density, 
          color=true_color, 
          label=true_label,
          linewidth=linewidth,
          linestyle=:dash)
    
    return p
end

"""
    plot_histogram_evolution_gif(tvec, xttraj, distribution; kwargs...)

Generates a GIF showing the evolution of histogram over time with the true distribution overlayed.

# Arguments
- `tvec`: Time vector of length T
- `xttraj`: Array of shape (state_dim, n_trajectories, n_timesteps) containing trajectories
- `distribution`: Distribution object from Distributions.jl (e.g., Normal, MixtureModel)
- `filename`: Output filename for the GIF (default: "histogram_evolution.gif")
- `fps`: Frames per second for the GIF (default: 10)
- `n_points`: Number of evaluation points for true density curve (default: 200)
- `xlim`: X-axis limits (default: automatic based on all data)
- `hist_color`: Color for histogram (default: :blue)
- `true_color`: Color for true density curve (default: :red)
- `hist_label`: Label for histogram (default: "Histogram")
- `true_label`: Label for true density curve (default: "True Density")
- `linewidth`: Line width for true density curve (default: 2)
- `alpha`: Transparency for histogram (default: 0.7)
- `bins`: Number of histogram bins (default: 30)
- `normalize`: Histogram normalization (:pdf, :density, :probability) (default: :pdf)
- `title_prefix`: Prefix for plot titles (default: "t = ")
- `every_n_frames`: Plot every nth frame to reduce file size (default: 1)

# Example
```julia
using Distributions
tvec = 0.0f0:0.01f0:1.0f0
xttraj = randn(Float32, 1, 1000, length(tvec))
true_dist = Normal(0, 1)
plot_histogram_evolution_gif(tvec, xttraj, true_dist, filename="evolution.gif")
```
"""
function plot_histogram_evolution_gif(tvec, xttraj, distribution; 
                                     filename="histogram_evolution.gif",
                                     fps=10,
                                     n_points=200,
                                     xlim=nothing,
                                     hist_color=:blue,
                                     true_color=:red,
                                     hist_label="Histogram",
                                     true_label="True Density",
                                     linewidth=2,
                                     alpha=0.7,
                                     bins=100,
                                     normalize=:pdf,
                                     title_prefix="t = ",
                                     every_n_frames=1)
    
    state_dim, n_trajectories, n_timesteps = size(xttraj)
    
    # For 1D state space
    if state_dim != 1
        error("Currently only supports 1D state space (state_dim = 1)")
    end
    
    # Determine x-axis limits from all data
    if xlim === nothing
        all_data = vec(xttraj[1, :, :])
        data_min, data_max = extrema(all_data)
        margin = 0.1 * (data_max - data_min)
        x_min = data_min - margin
        x_max = data_max + margin
    else
        x_min, x_max = xlim
    end
    
    # Create evaluation points for true density
    x_eval = range(x_min, x_max, length=n_points)
    
    # Compute true density once
    true_density = pdf.(distribution, x_eval)
    
    # Calculate freeze frames
    freeze_frames = Int(round(3 * fps))
    
    # Create animation with freeze frames
    anim = @animate for frame_idx in 1:(length(1:every_n_frames:n_timesteps) + freeze_frames)
        if frame_idx <= length(1:every_n_frames:n_timesteps)
            # Regular animation frames
            i = (frame_idx - 1) * every_n_frames + 1
            current_points = vec(xttraj[1, :, i])
            current_time = tvec[i]
        else
            # Freeze frames (last frame repeated)
            i = n_timesteps
            current_points = vec(xttraj[1, :, i])
            current_time = tvec[i]
        end
        
        # Create plot
        p = plot(xlabel="Value", 
                 ylabel="Density",
                 legend=:topright,
                 linewidth=linewidth,
                 title="$(title_prefix)$(round(current_time, digits=3))",
                 xlim=(x_min, x_max))
        
        # Plot histogram
        histogram!(p, current_points, 
                  alpha=alpha, 
                  color=hist_color, 
                  label=hist_label,
                  normalize=normalize,
                  bins=bins)
        
        # Plot true density
        plot!(p, x_eval, true_density, 
              color=true_color, 
              label=true_label,
              linewidth=linewidth,
              linestyle=:dash)
    end
    
    # Save as GIF
    gif(anim, filename, fps=fps)
    
    println("GIF saved as: $filename")
    return anim
end

"""
    plot_trajectories(conditional_path_times, conditional_path_states; kwargs...)

Plots trajectories from conditional path data structure where each trajectory has its own time and state vectors.

# Arguments
- `conditional_path_times`: Vector{Vector{Float32}} - Each inner vector contains time points for one trajectory
- `conditional_path_states`: Vector{Vector{SwitchState}} - Each inner vector contains SwitchState objects for one trajectory
- `n_plot`: Number of trajectories to plot (default: 50)
- `alpha`: Transparency of trajectory lines (default: 0.3)
- `linewidth`: Width of trajectory lines (default: 1)
- `color`: Color of trajectories (default: :blue)
- `legend`: Show legend (default: false)
- `extract_state`: Function to extract continuous state from SwitchState (default: x -> x.main_state.state[1,:])

# Example
```julia
# Assuming conditional_path_times and conditional_path_states are already populated
plot_trajectories(conditional_path_times, conditional_path_states)
```
"""
function plot_trajectories_conditional(conditional_path_times::Vector{Vector{Float32}}, 
                          conditional_path_states::Vector{Vector{T}}; 
                          n_plot=50,
                          alpha=0.3,
                          linewidth=1,
                          color=:blue,
                          legend=false,
                          extract_state = x -> x.main_state.state[1,:]) where T
    
    n_trajectories = length(conditional_path_times)
    
    # Validate that both vectors have the same length
    if length(conditional_path_states) != n_trajectories
        error("conditional_path_times and conditional_path_states must have the same length")
    end
    
    # Determine how many trajectories to plot
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
        times = conditional_path_times[idx]
        states = conditional_path_states[idx]
        
        # Extract continuous state values
        state_values = [extract_state(state) for state in states]
        
        # Convert to vectors if needed
        if length(state_values) > 0 && isa(state_values[1], AbstractVector)
            # If each state is a vector, take the first element (assuming 1D state space)
            state_values = [s[1] for s in state_values]
        end
        
        plot!(p, times, state_values, 
              alpha=alpha, 
              linewidth=linewidth,
              color=color,
              label="")
    end
    
    return p
end

# The original plot_trajectories function is now renamed to avoid conflicts
function plot_trajectories_standard(tvec, xttraj; 
                          n_plot=50,
                          alpha=0.3,
                          linewidth=1,
                          color=:blue,
                          legend=false,
                          x1_distribution=nothing,
                          x1_color=:red,
                          x1_n_points::Integer=200,
                          x1_scale::Real=0.05,
                          x1_label::AbstractString="X1 density")
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
    
    # Optionally overlay terminal distribution as vertical curve on the right
    if x1_distribution !== nothing
        all_data = vec(xttraj[1, :, :])
        y_min, y_max = extrema(all_data)
        y_eval = range(y_min, y_max, length=x1_n_points)
        dens = pdf.(x1_distribution, y_eval)
        maxd = maximum(dens)
        if maxd > 0
            t_end = maximum(tvec)
            x_offset = (maximum(tvec) - minimum(tvec)) * x1_scale
            x_curve = t_end .+ (dens ./ maxd) .* x_offset
            lab = legend ? x1_label : ""
            plot!(p, x_curve, y_eval; color=x1_color, linewidth=2, alpha=0.8, label=lab)
        end
    end

    return p
end

"""
    plot_trajectories(data...; kwargs...)

Convenience function that automatically dispatches to the appropriate plot_trajectories method
based on the input data structure.

# Arguments
- `data`: Either (tvec, xttraj) for standard format or (conditional_path_times, conditional_path_states) for conditional format
- `kwargs...`: Additional keyword arguments passed to the appropriate method

# Example
```julia
# Standard format
plot_trajectories(tvec, xttraj)

# Conditional format  
plot_trajectories(conditional_path_times, conditional_path_states)
```
"""
function plot_trajectories(data...; kwargs...)
    if length(data) == 2
        tvec, xttraj = data
        # Check if this is the conditional format
        if isa(tvec, Vector{Vector{Float32}}) && isa(xttraj, Vector{Vector})
            return plot_trajectories_conditional(tvec, xttraj; kwargs...)
        else
            # Assume standard format
            return plot_trajectories_standard(tvec, xttraj; kwargs...)
        end
    else
        error("plot_trajectories expects exactly 2 arguments")
    end
end

export plot_vector_endpoints, plot_trajectories, plot_histogram_vs_true_density, plot_histogram_evolution_gif

"""
    plot_quantile_mean_trajectories(tvec, xttraj; kwargs...)

Plots mean trajectories for percentile bins of trajectories, where binning is based on
each trajectory's final value (default). Bins are [0-2], [2-4], ..., [98-100] percentiles
by default, and one mean trajectory is plotted per bin.

# Arguments
- `tvec`: Time vector of length T
- `xttraj`: Array of shape (1, N, T) containing trajectories (1D state space)
- `percent_step`: Percentile step size (default: 2)
- `percent_min`: Minimum percentile edge (default: 0)
- `percent_max`: Maximum percentile edge (default: 100)
- `based_on`: Symbol for binning criterion, currently supports `:final` (default)
- `legend`: Show legend (default: false)
- `label_prefix`: Prefix for labels (default: "P")
- `line_color`: Color for all lines (default: :blue)
- `alpha_min`/`alpha_max`: Alpha range mapped by bin sample count (default: 0.3, 0.5)
- `linewidth_min`/`linewidth_max`: Line width range mapped by bin sample count (default: 0.5, 1.5)

# Example
```julia
plot_quantile_mean_trajectories(tvec, xttraj)
```
"""
function plot_quantile_mean_trajectories(tvec, xttraj; 
                                         percent_step::Integer=2,
                                         percent_min::Integer=0,
                                         percent_max::Integer=100,
                                         based_on::Symbol=:final,
                                         legend::Bool=false,
                                         label_prefix::AbstractString="P",
                                         line_color=:blue,
                                         alpha_min::Real=0.3,
                                         alpha_max::Real=0.5,
                                         linewidth_min::Real=0.5,
                                         linewidth_max::Real=1.5,
                                         x1_distribution=nothing,
                                         x1_color=:red,
                                         x1_n_points::Integer=200,
                                         x1_scale::Real=0.05,
                                         x1_label::AbstractString="X1 density")
    state_dim, n_trajectories, n_timesteps = size(xttraj)

    if state_dim != 1
        error("Currently only supports 1D state space (state_dim = 1)")
    end
    if n_trajectories == 0
        error("No trajectories provided (n_trajectories = 0)")
    end
    if percent_step <= 0 || percent_min < 0 || percent_max > 100 || percent_min >= percent_max
        error("Invalid percentile configuration")
    end

    # Determine binning scores per trajectory
    scores = if based_on == :final
        vec(xttraj[1, :, end])
    else
        error("Unsupported 'based_on' value: $(based_on). Only :final is supported.")
    end

    # Percentile edges and corresponding score thresholds
    edges_pct = collect(percent_min:percent_step:percent_max)
    if edges_pct[end] != percent_max
        push!(edges_pct, percent_max)
    end

    # Compute thresholds at edges using quantiles of the score distribution
    edges_thr = [quantile(scores, p/100) for p in edges_pct]

    # Prepare plot
    p = plot(xlabel="Time",
             ylabel="State",
             legend=legend)

    # For each bin between consecutive percentile edges, compute indices and counts
    n_bins = length(edges_thr) - 1
    bin_indices = Vector{Vector{Int}}(undef, n_bins)
    bin_counts = zeros(Int, n_bins)
    for i in 1:n_bins
        low_thr = edges_thr[i]
        high_thr = edges_thr[i+1]
        if i < n_bins
            mask = (scores .>= low_thr) .& (scores .< high_thr)
        else
            mask = (scores .>= low_thr) .& (scores .<= high_thr)
        end
        idxs = findall(mask)
        bin_indices[i] = idxs
        bin_counts[i] = length(idxs)
    end

    # Normalize counts for visual mapping
    nonempty_counts = filter(!=(0), bin_counts)
    if isempty(nonempty_counts)
        return p
    end
    minc = minimum(nonempty_counts)
    maxc = maximum(nonempty_counts)

    scale_count(c) = maxc == minc ? 1.0 : (c - minc) / (maxc - minc)

    # Plot mean trajectories with alpha and linewidth mapped by bin count
    for i in 1:n_bins
        idxs = bin_indices[i]
        isempty(idxs) && continue

        mean_traj = [mean(@view xttraj[1, idxs, t]) for t in 1:n_timesteps]
        r = scale_count(bin_counts[i])
        line_alpha = alpha_min + (alpha_max - alpha_min) * r
        line_width = linewidth_min + (linewidth_max - linewidth_min) * r

        label = string(label_prefix, edges_pct[i+1], "%")
        plot!(p, tvec, mean_traj; linewidth=line_width, alpha=line_alpha, color=line_color, label=label)
    end

    # Optionally overlay terminal distribution as vertical curve on the right
    if x1_distribution !== nothing
        all_data = vec(xttraj[1, :, :])
        y_min, y_max = extrema(all_data)
        y_eval = range(y_min, y_max, length=x1_n_points)
        dens = pdf.(x1_distribution, y_eval)
        maxd = maximum(dens)
        if maxd > 0
            t_end = maximum(tvec)
            x_offset = (maximum(tvec) - minimum(tvec)) * x1_scale
            x_curve = t_end .+ (dens ./ maxd) .* x_offset
            lab = legend ? x1_label : ""
            plot!(p, x_curve, y_eval; color=x1_color, linewidth=2, alpha=0.8, label=lab)
        end
    end

    return p
end

export plot_quantile_mean_trajectories

