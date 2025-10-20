default_maxdim() = typemax(Int)
default_mindim() = 1
default_cutoff() = 0.0

get_or_last(x, i::Integer) = (i >= length(x)) ? last(x) : x[i]

function truncation_parameters(
        sweep; cutoff = default_cutoff(), maxdim = default_maxdim(), mindim = default_mindim()
    )
    cutoff = get_or_last(cutoff, sweep)
    mindim = get_or_last(mindim, sweep)
    maxdim = get_or_last(maxdim, sweep)
    return (; cutoff, mindim, maxdim)
end
