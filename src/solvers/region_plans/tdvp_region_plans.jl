using Accessors: @modify

function applyexp_sub_steps(order)
    if order == 1
        return [1.0]
    elseif order == 2
        return [1 / 2, 1 / 2]
    elseif order == 4
        s = (2 - 2^(1 / 3))^(-1)
        return [s / 2, s / 2, 1 / 2 - s, 1 / 2 - s, s / 2, s / 2]
    else
        error("Applyexp order of $order not supported")
    end
end

function first_order_sweep(graph, sweep_kwargs; nsites)
    basic_fwd_sweep = post_order_dfs_plan(graph, sweep_kwargs; nsites)
    region_plan = []

    for (j, (region, region_kwargs)) in enumerate(basic_fwd_sweep)
        push!(region_plan, region => region_kwargs)

        if length(region) == 2 && j < length(basic_fwd_sweep)
            region_kwargs = @modify(-, region_kwargs.update!_kwargs.exponent_step)
            push!(region_plan, [last(region)] => region_kwargs)
        end
    end

    return region_plan
end

function reverse_regions(region_plan)
    return map(reverse(region_plan)) do (region, kwargs)
        return reverse(region) => kwargs
    end
end

# Generate the kwargs for each region.
function applyexp_regions(
        graph, raw_exponent_step; order, nsites, update!_kwargs = (; nsites), remaining_kwargs...
    )
    sweep_plan = []

    for (step, weight) in enumerate(applyexp_sub_steps(order))
        # Use this exponent step only if none provided
        new_update!_kwargs = (; exponent_step = weight * raw_exponent_step, update!_kwargs...)

        sweep_kwargs = (; remaining_kwargs..., update!_kwargs = new_update!_kwargs)

        region_plan = first_order_sweep(graph, sweep_kwargs; nsites)

        if iseven(step)
            region_plan = reverse_regions(region_plan)
        end

        append!(sweep_plan, region_plan)
    end

    return sweep_plan
end
