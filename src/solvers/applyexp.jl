using Printf: @printf

@kwdef mutable struct ApplyExpProblem{State} <: AbstractProblem
    operator
    state::State
    current_exponent::Number = 0.0
end

operator(A::ApplyExpProblem) = A.operator
state(A::ApplyExpProblem) = A.state
current_exponent(A::ApplyExpProblem) = A.current_exponent
function current_time(A::ApplyExpProblem)
    t = im * A.current_exponent
    return iszero(imag(t)) ? real(t) : t
end

# Rename region_plan
function region_plan(A::ApplyExpProblem; nsites, exponent_step, sweep_kwargs...)
    # The `exponent_step` kwarg for the `update!` function needs some pre-processing.
    return applyexp_regions(state(A), exponent_step; nsites, sweep_kwargs...)
end

function update!(
        region_iter::RegionIterator{<:ApplyExpProblem},
        local_state;
        nsites,
        exponent_step,
        solver = runge_kutta_solver,
    )
    prob = problem(region_iter)

    if iszero(abs(exponent_step))
        return region_iter, local_state
    end

    solver_kwargs = region_kwargs(solver, region_iter)

    local_state, _ = solver(
        x -> optimal_map(operator(prob), x), exponent_step, local_state; solver_kwargs...
    )
    if nsites == 1
        curr_reg = current_region(region_iter)
        next_reg = next_region(region_iter)
        if !isnothing(next_reg) && next_reg != curr_reg
            next_edge = first(edge_sequence_between_regions(state(prob), curr_reg, next_reg))
            v1, v2 = src(next_edge), dst(next_edge)
            psi = copy(state(prob))
            psi[v1], R = qr(local_state, uniqueinds(local_state, psi[v2]))
            shifted_operator = position(operator(prob), psi, NamedEdge(v1 => v2))
            R_t, _ = solver(
                x -> optimal_map(shifted_operator, x), -exponent_step, R; solver_kwargs...
            )
            local_state = psi[v1] * R_t
        end
    end

    prob.current_exponent += exponent_step

    return region_iter, local_state
end

function default_sweep_callback(
        sweep_iterator::SweepIterator{<:ApplyExpProblem};
        exponent_description = "exponent",
        outputlevel = 0,
        process_time = identity,
    )
    return if outputlevel >= 1
        the_problem = problem(sweep_iterator)
        @printf(
            "  Current %s = %s, ",
            exponent_description,
            process_time(current_exponent(the_problem))
        )
        @printf("maxlinkdim=%d", maxlinkdim(state(the_problem)))
        println()
        flush(stdout)
    end
end

function applyexp(
        init_prob::AbstractProblem,
        exponents;
        sweep_callback = default_sweep_callback,
        order = 4,
        nsites = 2,
        sweep_kwargs...,
    )
    exponent_steps = diff([zero(eltype(exponents)); exponents])

    kws_array = [
        (; order, nsites, sweep_kwargs..., exponent_step) for exponent_step in exponent_steps
    ]
    sweep_iter = SweepIterator(init_prob, kws_array)

    converged_prob = problem(sweep_solve!(sweep_callback, sweep_iter))

    return state(converged_prob)
end

function applyexp(operator, exponents, init_state; kws...)
    init_prob = ApplyExpProblem(;
        state = align_indices(init_state),
        operator = ProjTTN(align_indices(operator)),
        current_exponent = first(exponents),
    )
    return applyexp(init_prob, exponents; kws...)
end

process_real_times(z) = iszero(abs(z)) ? 0.0 : round(-imag(z); digits = 10)

function time_evolve(
        operator,
        time_points,
        init_state;
        process_time = process_real_times,
        sweep_callback = iter ->
        default_sweep_callback(iter; exponent_description = "time", process_time),
        sweep_kwargs...,
    )
    exponents = [-im * t for t in time_points]
    return applyexp(operator, exponents, init_state; sweep_callback, sweep_kwargs...)
end
