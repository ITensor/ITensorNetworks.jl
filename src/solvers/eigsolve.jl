using ITensors: truncerror
using Printf: @printf

@kwdef mutable struct EigsolveProblem{State, Operator} <: AbstractProblem
    operator::Operator
    state::State
    eigenvalue::Number = Inf
    max_truncerror::Real = 0.0
end

eigenvalue(E::EigsolveProblem) = E.eigenvalue
state(E::EigsolveProblem) = E.state
operator(E::EigsolveProblem) = E.operator
max_truncerror(E::EigsolveProblem) = E.max_truncerror

function set_truncation_info!(E::EigsolveProblem; spectrum = nothing)
    if !isnothing(spectrum)
        E.max_truncerror = max(max_truncerror(E), truncerror(spectrum))
    end
    return E
end

function update!(
        region_iter::RegionIterator{<:EigsolveProblem},
        local_state;
        solver = eigsolve_solver
    )
    prob = problem(region_iter)

    eigval, local_state = solver(
        ψ -> optimal_map(operator(prob), ψ), local_state;
        region_kwargs(solver, region_iter)...
    )

    prob.eigenvalue = eigval

    outputlevel = get(region_kwargs(region_iter), :outputlevel, 0)
    if outputlevel >= 2
        @printf(
            "  Region %s: energy = %.12f\n",
            current_region(region_iter),
            eigenvalue(prob)
        )
    end
    return region_iter, local_state
end

function default_sweep_callback(
        sweep_iterator::SweepIterator{<:EigsolveProblem}
    )
    outputlevel = get(region_kwargs(region_iterator(sweep_iterator)), :outputlevel, 0)
    return if outputlevel >= 1
        current_sweep = sweep_iterator.which_sweep
        the_problem = problem(sweep_iterator)
        @printf("After sweep %d ", current_sweep)
        @printf("eigenvalue=%.12f", eigenvalue(the_problem))
        @printf(" maxlinkdim=%d", maxlinkdim(state(the_problem)))
        @printf(" max truncerror=%d", max_truncerror(the_problem))
        println()
        flush(stdout)
    end
end

"""
    eigsolve(operator, init_state; nsweeps, nsites=1, factorize_kwargs, sweep_kwargs...) -> (eigenvalue, state)

Find the lowest eigenvalue and corresponding eigenvector of `operator` using a
DMRG-like sweep algorithm on a `TreeTensorNetwork`.

# Arguments

  - `operator`: The operator to diagonalize, typically a `TreeTensorNetwork` representing a
    Hamiltonian constructed from an `OpSum` (e.g. via `ttn(opsum, sites)`).
  - `init_state`: Initial guess for the eigenvector as a `TreeTensorNetwork`.
  - `nsweeps`: Number of sweeps over the network.
  - `nsites=1`: Number of sites optimized simultaneously per local update step (1 or 2).
  - `factorize_kwargs`: Keyword arguments controlling bond truncation after each local solve,
    e.g. `(; cutoff=1e-10, maxdim=50)`.
  - `outputlevel=0`: Level of output to print (0 = no output, 1 = sweep level information, 2 = step details)

# Returns

A tuple `(eigenvalue, state)` where `eigenvalue` is the converged lowest eigenvalue and
`state` is the optimized `TreeTensorNetwork` eigenvector.

# Example

```julia
energy, psi = eigsolve(H, psi0;
    nsweeps = 10,
    nsites = 2,
    factorize_kwargs = (; cutoff = 1e-10, maxdim = 50),
    outputlevel = 1
)
```

See also: [`dmrg`](@ref), [`time_evolve`](@ref).
"""
function eigsolve(
        operator, init_state; nsweeps, nsites = 1, factorize_kwargs, sweep_kwargs...
    )
    init_prob = EigsolveProblem(;
        state = align_indices(init_state), operator = ProjTTN(align_indices(operator))
    )
    sweep_iter = SweepIterator(
        init_prob,
        nsweeps;
        nsites,
        factorize_kwargs,
        subspace_expand!_kwargs = (; eigen_kwargs = factorize_kwargs),
        sweep_kwargs...
    )
    prob = problem(sweep_solve!(sweep_iter))
    return eigenvalue(prob), state(prob)
end

"""
    dmrg(operator, init_state; kwargs...) -> (eigenvalue, state)

Find the lowest eigenvalue and eigenvector of `operator` using the Density Matrix
Renormalization Group (DMRG) algorithm. This is an alias for [`eigsolve`](@ref).

See [`eigsolve`](@ref) for the full description of arguments and keyword arguments.

# Example

```julia
energy, psi = dmrg(H, psi0;
    nsweeps = 10,
    nsites = 2,
    factorize_kwargs = (; cutoff = 1e-10, maxdim = 50)
)
```
"""
dmrg(operator, init_state; kwargs...) = eigsolve(operator, init_state; kwargs...)
