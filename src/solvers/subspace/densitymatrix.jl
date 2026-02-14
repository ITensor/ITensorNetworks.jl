using NamedGraphs.GraphsExtensions: incident_edges
using Printf: @printf

function subspace_expand!(
        ::Backend"densitymatrix",
        region_iter,
        local_state;
        expansion_factor = 1.5,
        maxexpand = typemax(Int),
        north_pass = 1,
        eigen_kwargs = (;)
    )
    prob = problem(region_iter)

    region = current_region(region_iter)
    psi = copy(state(prob))

    prev_vertex_set = setdiff(pos(operator(prob)), region)
    (length(prev_vertex_set) != 1) && return region_iter, local_state
    prev_vertex = only(prev_vertex_set)
    A = psi[prev_vertex]

    next_vertices = filter(v -> (hascommoninds(psi[v], A)), region)
    isempty(next_vertices) && return region_iter, local_state
    next_vertex = only(next_vertices)
    C = psi[next_vertex]

    a = commonind(A, C)
    isnothing(a) && return region_iter, local_state
    basis_size = prod(dim.(uniqueinds(A, C)))

    trunc_kwargs = truncation_parameters(region_iter.which_sweep; eigen_kwargs...)
    expanded_maxdim = compute_expansion(
        dim(a), basis_size; expansion_factor, maxexpand, trunc_kwargs.maxdim
    )
    expanded_maxdim <= 0 && return region_iter, local_state

    envs = environments(operator(prob))
    H = operator(operator(prob))
    sqrt_rho = A
    for e in incident_edges(operator(prob))
        (src(e) ∈ region || dst(e) ∈ region) && continue
        sqrt_rho *= envs[e]
    end
    sqrt_rho *= H[prev_vertex]

    conj_proj_A(T) = (T - prime(A) * (dag(prime(A)) * T))
    for _ in 1:north_pass
        sqrt_rho = conj_proj_A(sqrt_rho)
    end
    rho = sqrt_rho * dag(noprime(sqrt_rho))
    D, U = eigen(rho; trunc_kwargs..., ishermitian = true)

    Uproj(T) = (T - prime(A, a) * (dag(prime(A, a)) * T))
    for _ in 1:north_pass
        U = Uproj(U)
    end
    if norm(dag(U) * A) > 1.0e-10
        @printf("Warning: |U*A| = %.3E in subspace expansion\n", norm(dag(U) * A))
        return region_iter, local_state
    end

    Ax, ax = directsum(A => a, U => commonind(U, D))
    expander = dag(Ax) * A
    psi[prev_vertex] = Ax
    psi[next_vertex] = expander * C
    local_state = expander * local_state

    prob.state = psi

    return region_iter, local_state
end
