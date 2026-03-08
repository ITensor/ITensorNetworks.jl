using .BaseExtensions: maybe_real
using Graphs: has_edge
using ITensors: ITensors, ITensor, Index, Ops, apply, commonind, commoninds, contract, dag,
    denseblocks, factorize, factorize_svd, hasqns, isdiag, noprime, prime, replaceind,
    replaceinds, unioninds, uniqueinds
using KrylovKit: linsolve
using LinearAlgebra: eigen, norm, qr, svd
using NamedGraphs: NamedEdge, has_edge

function full_update_bp(
        o::Union{NamedEdge, ITensor},
        ψ,
        v⃗;
        envs,
        nfullupdatesweeps = 10,
        print_fidelity_loss = false,
        envisposdef = false,
        callback = Returns(nothing),
        symmetrize = false,
        apply_kwargs...
    )
    outer_dim_v1, outer_dim_v2 = dim(uniqueinds(ψ[v⃗[1]], o, ψ[v⃗[2]])),
        dim(uniqueinds(ψ[v⃗[2]], o, ψ[v⃗[1]]))
    dim_shared = dim(commoninds(ψ[v⃗[1]], ψ[v⃗[2]]))
    d1, d2 = dim(commoninds(ψ[v⃗[1]], o)), dim(commoninds(ψ[v⃗[2]], o))
    if outer_dim_v1 * outer_dim_v2 <= dim_shared * dim_shared * d1 * d2
        Qᵥ₁, Rᵥ₁ = ITensor(true), copy(ψ[v⃗[1]])
        Qᵥ₂, Rᵥ₂ = ITensor(true), copy(ψ[v⃗[2]])
    else
        Qᵥ₁, Rᵥ₁ = factorize(
            ψ[v⃗[1]], uniqueinds(uniqueinds(ψ[v⃗[1]], ψ[v⃗[2]]), uniqueinds(ψ, v⃗[1]))
        )
        Qᵥ₂, Rᵥ₂ = factorize(
            ψ[v⃗[2]], uniqueinds(uniqueinds(ψ[v⃗[2]], ψ[v⃗[1]]), uniqueinds(ψ, v⃗[2]))
        )
    end
    extended_envs = vcat(envs, Qᵥ₁, prime(dag(Qᵥ₁)), Qᵥ₂, prime(dag(Qᵥ₂)))
    Rᵥ₁, Rᵥ₂ = optimise_p_q(
        Rᵥ₁,
        Rᵥ₂,
        extended_envs,
        o;
        nfullupdatesweeps,
        print_fidelity_loss,
        envisposdef,
        apply_kwargs...
    )
    if symmetrize
        singular_values! = Ref(ITensor())
        Rᵥ₁, Rᵥ₂, spec = factorize_svd(
            Rᵥ₁ * Rᵥ₂,
            inds(Rᵥ₁);
            ortho = "none",
            tags = edge_tag(v⃗[1] => v⃗[2]),
            singular_values!,
            apply_kwargs...
        )
        callback(; singular_values = singular_values![], truncation_error = spec.truncerr)
    end
    ψᵥ₁ = Qᵥ₁ * Rᵥ₁
    ψᵥ₂ = Qᵥ₂ * Rᵥ₂
    return ψᵥ₁, ψᵥ₂
end

function simple_update_bp_full(
        o::Union{NamedEdge, ITensor}, ψ, v⃗; envs, callback = Returns(nothing), apply_kwargs...
    )
    cutoff = 10 * eps(real(scalartype(ψ)))
    envs_v1 = filter(env -> hascommoninds(env, ψ[v⃗[1]]), envs)
    envs_v2 = filter(env -> hascommoninds(env, ψ[v⃗[2]]), envs)
    @assert all(ndims(env) == 2 for env in vcat(envs_v1, envs_v2))
    sqrt_envs_v1 = [
        ITensorsExtensions.map_eigvals(
                sqrt, env, inds(env)[1], inds(env)[2]; cutoff, ishermitian = true
            ) for env in envs_v1
    ]
    sqrt_envs_v2 = [
        ITensorsExtensions.map_eigvals(
                sqrt, env, inds(env)[1], inds(env)[2]; cutoff, ishermitian = true
            ) for env in envs_v2
    ]
    inv_sqrt_envs_v1 = [
        ITensorsExtensions.map_eigvals(
                inv ∘ sqrt, env, inds(env)[1], inds(env)[2]; cutoff, ishermitian = true
            ) for env in envs_v1
    ]
    inv_sqrt_envs_v2 = [
        ITensorsExtensions.map_eigvals(
                inv ∘ sqrt, env, inds(env)[1], inds(env)[2]; cutoff, ishermitian = true
            ) for env in envs_v2
    ]
    ψᵥ₁ᵥ₂_tn = [ψ[v⃗[1]]; ψ[v⃗[2]]; sqrt_envs_v1; sqrt_envs_v2]
    ψᵥ₁ᵥ₂ = contract(ψᵥ₁ᵥ₂_tn; sequence = contraction_sequence(ψᵥ₁ᵥ₂_tn; alg = "optimal"))
    oψ = apply(o, ψᵥ₁ᵥ₂)
    v1_inds = reduce(
        vcat, [uniqueinds(sqrt_env_v1, ψ[v⃗[1]]) for sqrt_env_v1 in sqrt_envs_v1];
        init = Index[]
    )
    v2_inds = reduce(
        vcat, [uniqueinds(sqrt_env_v2, ψ[v⃗[2]]) for sqrt_env_v2 in sqrt_envs_v2];
        init = Index[]
    )
    v1_inds = [v1_inds; siteinds(ψ, v⃗[1])]
    v2_inds = [v2_inds; siteinds(ψ, v⃗[2])]
    e = v⃗[1] => v⃗[2]
    singular_values! = Ref(ITensor())
    ψᵥ₁, ψᵥ₂, spec = factorize_svd(
        oψ, v1_inds; ortho = "none", tags = edge_tag(e), singular_values!,
        apply_kwargs...
    )
    callback(; singular_values = singular_values![], truncation_error = spec.truncerr)
    for inv_sqrt_env_v1 in inv_sqrt_envs_v1
        ψᵥ₁ *= dag(inv_sqrt_env_v1)
    end
    for inv_sqrt_env_v2 in inv_sqrt_envs_v2
        ψᵥ₂ *= dag(inv_sqrt_env_v2)
    end
    return ψᵥ₁, ψᵥ₂
end

# Reduced version
function simple_update_bp(
        o::Union{NamedEdge, ITensor}, ψ, v⃗; envs, callback = Returns(nothing), apply_kwargs...
    )
    cutoff = 10 * eps(real(scalartype(ψ)))
    envs_v1 = filter(env -> hascommoninds(env, ψ[v⃗[1]]), envs)
    envs_v2 = filter(env -> hascommoninds(env, ψ[v⃗[2]]), envs)
    @assert all(ndims(env) == 2 for env in vcat(envs_v1, envs_v2))
    sqrt_envs_v1 = [
        ITensorsExtensions.map_eigvals(
                sqrt, env, inds(env)[1], inds(env)[2]; cutoff, ishermitian = true
            ) for env in envs_v1
    ]
    sqrt_envs_v2 = [
        ITensorsExtensions.map_eigvals(
                sqrt, env, inds(env)[1], inds(env)[2]; cutoff, ishermitian = true
            ) for env in envs_v2
    ]
    inv_sqrt_envs_v1 = [
        ITensorsExtensions.map_eigvals(
                inv ∘ sqrt, env, inds(env)[1], inds(env)[2]; cutoff, ishermitian = true
            ) for env in envs_v1
    ]
    inv_sqrt_envs_v2 = [
        ITensorsExtensions.map_eigvals(
                inv ∘ sqrt, env, inds(env)[1], inds(env)[2]; cutoff, ishermitian = true
            ) for env in envs_v2
    ]
    ψᵥ₁ = contract([ψ[v⃗[1]]; sqrt_envs_v1])
    ψᵥ₂ = contract([ψ[v⃗[2]]; sqrt_envs_v2])
    sᵥ₁ = siteinds(ψ, v⃗[1])
    sᵥ₂ = siteinds(ψ, v⃗[2])
    Qᵥ₁, Rᵥ₁ = qr(ψᵥ₁, uniqueinds(uniqueinds(ψᵥ₁, ψᵥ₂), sᵥ₁))
    Qᵥ₂, Rᵥ₂ = qr(ψᵥ₂, uniqueinds(uniqueinds(ψᵥ₂, ψᵥ₁), sᵥ₂))
    rᵥ₁ = commoninds(Qᵥ₁, Rᵥ₁)
    rᵥ₂ = commoninds(Qᵥ₂, Rᵥ₂)
    oR = apply(o, Rᵥ₁ * Rᵥ₂)
    e = v⃗[1] => v⃗[2]
    singular_values! = Ref(ITensor())
    Rᵥ₁, Rᵥ₂, spec = factorize_svd(
        oR,
        unioninds(rᵥ₁, sᵥ₁);
        ortho = "none",
        tags = edge_tag(e),
        singular_values!,
        apply_kwargs...
    )
    callback(; singular_values = singular_values![], truncation_error = spec.truncerr)
    Qᵥ₁ = contract([Qᵥ₁; dag.(inv_sqrt_envs_v1)])
    Qᵥ₂ = contract([Qᵥ₂; dag.(inv_sqrt_envs_v2)])
    ψᵥ₁ = Qᵥ₁ * Rᵥ₁
    ψᵥ₂ = Qᵥ₂ * Rᵥ₂
    return ψᵥ₁, ψᵥ₂
end

function ITensors.apply(
        o::Union{NamedEdge, ITensor},
        ψ::AbstractITensorNetwork;
        envs = ITensor[],
        normalize = false,
        ortho = false,
        nfullupdatesweeps = 10,
        print_fidelity_loss = false,
        envisposdef = false,
        callback = Returns(nothing),
        variational_optimization_only = false,
        symmetrize = false,
        reduced = true,
        apply_kwargs...
    )
    ψ = copy(ψ)
    v⃗ = neighbor_vertices(ψ, o)
    if length(v⃗) == 1
        if ortho
            ψ = tree_orthogonalize(ψ, v⃗[1])
        end
        oψᵥ = apply(o, ψ[v⃗[1]])
        if normalize
            oψᵥ ./= norm(oψᵥ)
        end
        setindex_preserve_graph!(ψ, oψᵥ, v⃗[1])
    elseif length(v⃗) == 2
        envs = Vector{ITensor}(envs)
        is_product_env = iszero(ne(ITensorNetwork(envs)))
        e = v⃗[1] => v⃗[2]
        if !has_edge(ψ, e)
            error("Vertices where the gates are being applied must be neighbors for now.")
        end
        if ortho
            ψ = tree_orthogonalize(ψ, v⃗[1])
        end
        if variational_optimization_only || !is_product_env
            ψᵥ₁, ψᵥ₂ = full_update_bp(
                o,
                ψ,
                v⃗;
                envs,
                nfullupdatesweeps,
                print_fidelity_loss,
                envisposdef,
                callback,
                symmetrize,
                apply_kwargs...
            )
        else
            if reduced
                ψᵥ₁, ψᵥ₂ = simple_update_bp(o, ψ, v⃗; envs, callback, apply_kwargs...)
            else
                ψᵥ₁, ψᵥ₂ = simple_update_bp_full(o, ψ, v⃗; envs, callback, apply_kwargs...)
            end
        end
        if normalize
            ψᵥ₁ ./= norm(ψᵥ₁)
            ψᵥ₂ ./= norm(ψᵥ₂)
        end
        setindex_preserve_graph!(ψ, ψᵥ₁, v⃗[1])
        setindex_preserve_graph!(ψ, ψᵥ₂, v⃗[2])
    elseif length(v⃗) < 1
        error("Gate being applied does not share indices with tensor network.")
    elseif length(v⃗) > 2
        error("Gates with more than 2 sites is not supported yet.")
    end
    return ψ
end

function ITensors.apply(
        o⃗::Union{Vector{NamedEdge}, Vector{ITensor}},
        ψ::AbstractITensorNetwork;
        normalize = false,
        ortho = false,
        apply_kwargs...
    )
    o⃗ψ = ψ
    for oᵢ in o⃗
        o⃗ψ = apply(oᵢ, o⃗ψ; normalize, ortho, apply_kwargs...)
    end
    return o⃗ψ
end

function ITensors.apply(
        o⃗::Scaled,
        ψ::AbstractITensorNetwork;
        cutoff = nothing,
        normalize = false,
        ortho = false,
        apply_kwargs...
    )
    return maybe_real(Ops.coefficient(o⃗)) *
        apply(Ops.argument(o⃗), ψ; cutoff, maxdim, normalize, ortho, apply_kwargs...)
end

function ITensors.apply(
        o⃗::Prod, ψ::AbstractITensorNetwork; normalize = false, ortho = false, apply_kwargs...
    )
    o⃗ψ = ψ
    for oᵢ in o⃗
        o⃗ψ = apply(oᵢ, o⃗ψ; normalize, ortho, apply_kwargs...)
    end
    return o⃗ψ
end

function ITensors.apply(
        o::Op, ψ::AbstractITensorNetwork; normalize = false, ortho = false, apply_kwargs...
    )
    return apply(ITensor(o, siteinds(ψ)), ψ; normalize, ortho, apply_kwargs...)
end

_gate_vertices(o::ITensor, ψ) = neighbor_vertices(ψ, o)
_gate_vertices(o::AbstractEdge, ψ) = [src(o), dst(o)]

function _contract_gate(o::ITensor, ψv1, Λ, ψv2)
    indsᵥ₁ = noprime(noncommoninds(ψv1, Λ))
    Qᵥ₁, Rᵥ₁ = qr(ψv1, setdiff(uniqueinds(indsᵥ₁, ψv2), commoninds(indsᵥ₁, o)))
    Qᵥ₂, Rᵥ₂ = qr(ψv2, setdiff(uniqueinds(ψv2, indsᵥ₁), commoninds(ψv2, o)))
    theta = noprime(noprime(Rᵥ₁ * Λ) * Rᵥ₂ * o)
    return Qᵥ₁, Rᵥ₁, Qᵥ₂, Rᵥ₂, theta
end

function _contract_gate(o::AbstractEdge, ψv1, Λ, ψv2)
    indsᵥ₁ = noprime(noncommoninds(ψv1, Λ))
    Qᵥ₁, Rᵥ₁ = qr(ψv1, uniqueinds(indsᵥ₁, ψv2))
    Qᵥ₂, Rᵥ₂ = qr(ψv2, uniqueinds(ψv2, indsᵥ₁))
    theta = noprime(Rᵥ₁ * Λ) * Rᵥ₂
    return Qᵥ₁, Rᵥ₁, Qᵥ₂, Rᵥ₂, theta
end

# In the future we will try to unify this into apply() above but currently leave it mostly as a separate function
# Apply() function for an ITN in the Vidal Gauge. Hence the bond tensors are required.
# Gate does not necessarily need to be passed. Can supply an edge to do an identity update instead. Uses Simple Update procedure assuming gate is two-site
function ITensors.apply(
        o::Union{NamedEdge, ITensor}, ψ::VidalITensorNetwork; normalize = false, apply_kwargs...
    )
    updated_ψ = copy(site_tensors(ψ))
    updated_bond_tensors = copy(bond_tensors(ψ))
    v⃗ = _gate_vertices(o, ψ)
    if length(v⃗) == 2
        e = NamedEdge(v⃗[1] => v⃗[2])
        ψv1, ψv2 = ψ[src(e)], ψ[dst(e)]
        e_ind = commonind(ψv1, ψv2)

        for vn in neighbors(ψ, src(e))
            if (vn != dst(e))
                ψv1 = noprime(ψv1 * bond_tensor(ψ, vn => src(e)))
            end
        end

        for vn in neighbors(ψ, dst(e))
            if (vn != src(e))
                ψv2 = noprime(ψv2 * bond_tensor(ψ, vn => dst(e)))
            end
        end

        Qᵥ₁, Rᵥ₁, Qᵥ₂, Rᵥ₂, theta = _contract_gate(o, ψv1, bond_tensor(ψ, e), ψv2)

        U, S, V = ITensors.svd(
            theta,
            uniqueinds(Rᵥ₁, Rᵥ₂);
            lefttags = ITensorNetworks.edge_tag(e),
            righttags = ITensorNetworks.edge_tag(e),
            apply_kwargs...
        )

        ind_to_replace = commonind(V, S)
        ind_to_replace_with = commonind(U, S)
        S = replaceind(S, ind_to_replace => ind_to_replace_with')
        V = replaceind(V, ind_to_replace => ind_to_replace_with)

        ψv1, updated_bond_tensors[e], ψv2 = U * Qᵥ₁, S, V * Qᵥ₂

        for vn in neighbors(ψ, src(e))
            if (vn != dst(e))
                ψv1 =
                    noprime(ψv1 * ITensorsExtensions.inv_diag(bond_tensor(ψ, vn => src(e))))
            end
        end

        for vn in neighbors(ψ, dst(e))
            if (vn != src(e))
                ψv2 =
                    noprime(ψv2 * ITensorsExtensions.inv_diag(bond_tensor(ψ, vn => dst(e))))
            end
        end

        if normalize
            ψv1 /= norm(ψv1)
            ψv2 /= norm(ψv2)
            updated_bond_tensors[e] /= norm(updated_bond_tensors[e])
        end

        setindex_preserve_graph!(updated_ψ, ψv1, src(e))
        setindex_preserve_graph!(updated_ψ, ψv2, dst(e))

        return VidalITensorNetwork(updated_ψ, updated_bond_tensors)

    else
        updated_ψ = apply(o, updated_ψ; normalize)
        return VidalITensorNetwork(updated_ψ, updated_bond_tensors)
    end
end

### Full Update Routines ###

# Calculate the overlap of the gate acting on the previous p and q versus the new p and q in the presence of environments. This is the cost function that optimise_p_q will minimise
function fidelity(
        envs::Vector{ITensor},
        p_cur::ITensor,
        q_cur::ITensor,
        p_prev::ITensor,
        q_prev::ITensor,
        gate::ITensor
    )
    p_sind, q_sind = commonind(p_cur, gate), commonind(q_cur, gate)
    p_sind_sim, q_sind_sim = sim(p_sind), sim(q_sind)
    gate_sq =
        gate * replaceinds(dag(gate), Index[p_sind, q_sind], Index[p_sind_sim, q_sind_sim])
    term1_tns = vcat(
        [
            p_prev,
            q_prev,
            replaceind(prime(dag(p_prev)), prime(p_sind), p_sind_sim),
            replaceind(prime(dag(q_prev)), prime(q_sind), q_sind_sim),
            gate_sq,
        ],
        envs
    )
    sequence = contraction_sequence(term1_tns; alg = "optimal")
    term1 = ITensors.contract(term1_tns; sequence)

    term2_tns = vcat(
        [
            p_cur,
            q_cur,
            replaceind(prime(dag(p_cur)), prime(p_sind), p_sind),
            replaceind(prime(dag(q_cur)), prime(q_sind), q_sind),
        ],
        envs
    )
    sequence = contraction_sequence(term2_tns; alg = "optimal")
    term2 = ITensors.contract(term2_tns; sequence)
    term3_tns = vcat([p_prev, q_prev, prime(dag(p_cur)), prime(dag(q_cur)), gate], envs)
    sequence = contraction_sequence(term3_tns; alg = "optimal")
    term3 = ITensors.contract(term3_tns; sequence)

    f = term3[] / sqrt(term1[] * term2[])
    return f * conj(f)
end

# Do Full Update Sweeping, Optimising the tensors p and q in the presence of the environments envs,
# Specifically this functions find the p_cur and q_cur which optimise envs*gate*p*q*dag(prime(p_cur))*dag(prime(q_cur))
function optimise_p_q(
        p::ITensor,
        q::ITensor,
        envs::Vector{ITensor},
        o::ITensor;
        nfullupdatesweeps = 10,
        print_fidelity_loss = false,
        envisposdef = true,
        apply_kwargs...
    )
    p_cur, q_cur = factorize(
        apply(o, p * q), inds(p); tags = tags(commonind(p, q)), apply_kwargs...
    )

    fstart = print_fidelity_loss ? fidelity(envs, p_cur, q_cur, p, q, o) : 0

    qs_ind = setdiff(inds(q_cur), collect(Iterators.flatten(inds.(vcat(envs, p_cur)))))
    ps_ind = setdiff(inds(p_cur), collect(Iterators.flatten(inds.(vcat(envs, q_cur)))))

    function b(p::ITensor, q::ITensor, o::ITensor, envs::Vector{ITensor}, r::ITensor)
        ts = vcat(ITensor[p, q, o, dag(prime(r))], envs)
        sequence = contraction_sequence(ts; alg = "optimal")
        return noprime(ITensors.contract(ts; sequence))
    end

    function M_p(envs::Vector{ITensor}, p_q_tensor::ITensor, s_ind, apply_tensor::ITensor)
        ts = vcat(
            ITensor[
                p_q_tensor, replaceinds(prime(dag(p_q_tensor)), prime(s_ind), s_ind),
                apply_tensor,
            ],
            envs
        )
        sequence = contraction_sequence(ts; alg = "optimal")
        return noprime(ITensors.contract(ts; sequence))
    end
    for i in 1:nfullupdatesweeps
        b_vec = b(p, q, o, envs, q_cur)
        M_p_partial = partial(M_p, envs, q_cur, qs_ind)

        p_cur, info = linsolve(
            M_p_partial, b_vec, p_cur; isposdef = envisposdef, ishermitian = false
        )

        b_tilde_vec = b(p, q, o, envs, p_cur)
        M_p_tilde_partial = partial(M_p, envs, p_cur, ps_ind)

        q_cur, info = linsolve(
            M_p_tilde_partial, b_tilde_vec, q_cur; isposdef = envisposdef,
            ishermitian = false
        )
    end

    fend = print_fidelity_loss ? fidelity(envs, p_cur, q_cur, p, q, o) : 0

    diff = real(fend - fstart)
    if print_fidelity_loss && diff < -eps(diff) && nfullupdatesweeps >= 1
        println(
            "Warning: Krylov Solver Didn't Find a Better Solution by Sweeping. Something might be amiss."
        )
    end

    return p_cur, q_cur
end

partial = (f, a...; c...) -> (b...) -> f(a..., b...; c...)
