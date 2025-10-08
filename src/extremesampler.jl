function generate_extreme_sample(d::Distribution, p)
    proposal = rand(d)
    if cdf(d, proposal) < p || cdf(d, proposal) > 1 - p
        return proposal
    end
    return generate_extreme_sample(d, p)
end
function extreme_matrix_sample(d::Distribution, p, n)
    sample = fill(0.0,1,n)
    for i in 1:n
        sample[1,i] = generate_extreme_sample(d, p)
    end
    return sample
end