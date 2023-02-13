# function ITensors.apply(
#   o::ITensor,
#   ψ::AbstractITensorNetwork;
#   cutoff=nothing,
#   maxdim=nothing,
#   normalize=false,
#   ortho=false,
# )
#   ψ = copy(ψ)
#   v⃗ = neighbor_vertices(ψ, o)
#   if length(v⃗) == 1
#     if ortho
#       ψ = orthogonalize(ψ, v⃗[1])
#     end
#     oψᵥ = apply(o, ψ[v⃗[1]])
#     if normalize
#       oψᵥ ./= norm(oψᵥ)
#     end
#     ψ[v⃗[1]] = oψᵥ
#   elseif length(v⃗) == 2
#     e = v⃗[1] => v⃗[2]
#     if !has_edge(ψ, e)
#       error("Vertices where the gates are being applied must be neighbors for now.")
#     end
#     if ortho
#       ψ = orthogonalize(ψ, v⃗[1])
#     end

#     #Check whether to do a memory efficient QR first (do both sites combined for now, although we could split if we wish)
#     #if no maxdim is provided just do it based on inds counting, if it is provided do it based on whichever takes less memory

#     dim_v1 = dim(uniqueinds(ψ[v⃗[1]], ψ[v⃗[2]]))
#     dim_v2 = dim(uniqueinds(ψ[v⃗[2]], ψ[v⃗[1]]))
#     dim_shared = dim(commoninds(ψ[v⃗[1]], ψ[v⃗[2]]))
#     d1, d2 = dim(uniqueinds(ψ, v⃗[1])), dim(uniqueinds(ψ, v⃗[2]))
#     if dim_v1*dim_v2 <= d1*d2*dim_shared*dim_shared*d1*d2
#       oψᵥ = apply(o, ψ[v⃗[1]] * ψ[v⃗[2]])
#       ψᵥ₁, ψᵥ₂ = factorize(
#         oψᵥ, inds(ψ[v⃗[1]]); cutoff, maxdim, tags=ITensorNetworks.edge_tag(e)
#       )
#     else
#       Qv1, Rv1 = factorize(
#         ψ[v⃗[1]], uniqueinds(uniqueinds(ψ[v⃗[1]], ψ[v⃗[2]]), uniqueinds(ψ, v⃗[1]))
#       )
#       Qv2, Rv2 = factorize(
#         ψ[v⃗[2]], uniqueinds(uniqueinds(ψ[v⃗[2]], ψ[v⃗[1]]), uniqueinds(ψ, v⃗[2]))
#       )
#       Rv1_new, Rv2_new = factorize(
#         noprime(Rv1 * o * Rv2), inds(Rv1); cutoff, maxdim, tags=ITensorNetworks.edge_tag(e)
#       )
#       ψᵥ₁ = Qv1 * Rv1_new
#       ψᵥ₂ = Qv2 * Rv2_new
#     end

#     if normalize
#       ψᵥ₁ ./= norm(ψᵥ₁)
#       ψᵥ₂ ./= norm(ψᵥ₂)
#     end

#     ψ[v⃗[1]] = ψᵥ₁
#     ψ[v⃗[2]] = ψᵥ₂

#   elseif length(v⃗) < 1
#     error("Gate being applied does not share indices with tensor network.")
#   elseif length(v⃗) > 2
#     error("Gates with more than 2 sites is not supported yet.")
#   end
#   return ψ
# end

function ITensors.apply(
  o::ITensor,
  ψ::AbstractITensorNetwork;
  cutoff=nothing,
  maxdim=nothing,
  normalize=false,
  ortho=false,
  envs=ITensor[],
  nfullupdatesweeps=10,
  print_fidelity_loss=true,
  envisposdef=false,
)
  ψ = copy(ψ)
  v⃗ = neighbor_vertices(ψ, o)
  if length(v⃗) == 1
    if ortho
      ψ = orthogonalize(ψ, v⃗[1])
    end
    oψᵥ = apply(o, ψ[v⃗[1]])
    if normalize
      oψᵥ ./= norm(oψᵥ)
    end
    ψ[v⃗[1]] = oψᵥ
  elseif length(v⃗) == 2
    e = v⃗[1] => v⃗[2]
    if !has_edge(ψ, e)
      error("Vertices where the gates are being applied must be neighbors for now.")
    end
    if ortho
      ψ = orthogonalize(ψ, v⃗[1])
    end

    outer_dim_v1, outer_dim_v2 = dim(uniqueinds(ψ[v⃗[1]], o, ψ[v⃗[2]])),
    dim(uniqueinds(ψ[v⃗[2]], o, ψ[v⃗[1]]))
    dim_shared = dim(commoninds(ψ[v⃗[1]], ψ[v⃗[2]]))
    d1, d2 = dim(commoninds(ψ[v⃗[1]], o)), dim(commoninds(ψ[v⃗[2]], o))
    if outer_dim_v1 * outer_dim_v2 <= dim_shared * dim_shared * d1 * d2
      Qᵥ₁, Rᵥ₁ = ITensor(1.0), copy(ψ[v⃗[1]])
      Qᵥ₂, Rᵥ₂ = ITensor(1.0), copy(ψ[v⃗[2]])
    else
      Qᵥ₁, Rᵥ₁ = factorize(
        ψ[v⃗[1]], uniqueinds(uniqueinds(ψ[v⃗[1]], ψ[v⃗[2]]), uniqueinds(ψ, v⃗[1]))
      )
      Qᵥ₂, Rᵥ₂ = factorize(
        ψ[v⃗[2]], uniqueinds(uniqueinds(ψ[v⃗[2]], ψ[v⃗[1]]), uniqueinds(ψ, v⃗[2]))
      )
    end

    if !isempty(envs)
      extended_envs = vcat(envs, Qᵥ₁, prime(dag(Qᵥ₁)), Qᵥ₂, prime(dag(Qᵥ₂)))
      Rᵥ₁, Rᵥ₂ = optimise_p_q(
        Rᵥ₁,
        Rᵥ₂,
        extended_envs,
        o;
        nfullupdatesweeps,
        maxdim,
        print_fidelity_loss,
        envisposdef,
      )
    else
      Rᵥ₁, Rᵥ₂ = factorize(
        apply(o, Rᵥ₁ * Rᵥ₂), inds(Rᵥ₁); cutoff, maxdim, tags=ITensorNetworks.edge_tag(e)
      )
    end

    ψᵥ₁ = Qᵥ₁ * Rᵥ₁
    ψᵥ₂ = Qᵥ₂ * Rᵥ₂

    if normalize
      ψᵥ₁ ./= norm(ψᵥ₁)
      ψᵥ₂ ./= norm(ψᵥ₂)
    end

    ψ[v⃗[1]] = ψᵥ₁
    ψ[v⃗[2]] = ψᵥ₂

  elseif length(v⃗) < 1
    error("Gate being applied does not share indices with tensor network.")
  elseif length(v⃗) > 2
    error("Gates with more than 2 sites is not supported yet.")
  end
  return ψ
end

function ITensors.apply(
  o⃗::Vector{ITensor},
  ψ::AbstractITensorNetwork;
  cutoff,
  maxdim=typemax(Int),
  normalize=false,
  ortho=false,
)
  o⃗ψ = ψ
  for oᵢ in o⃗
    o⃗ψ = apply(oᵢ, o⃗ψ; cutoff, maxdim, normalize, ortho)
  end
  return o⃗ψ
end

function ITensors.apply(
  o⃗::Scaled, ψ::AbstractITensorNetwork; cutoff, maxdim, normalize=false, ortho=false
)
  return maybe_real(Ops.coefficient(o⃗)) *
         apply(Ops.argument(o⃗), ψ; cutoff, maxdim, normalize, ortho)
end

function ITensors.apply(
  o⃗::Prod, ψ::AbstractITensorNetwork; cutoff, maxdim, normalize=false, ortho=false
)
  o⃗ψ = ψ
  for oᵢ in o⃗
    o⃗ψ = apply(oᵢ, o⃗ψ; cutoff, maxdim, normalize, ortho)
  end
  return o⃗ψ
end

function ITensors.apply(
  o::Op, ψ::AbstractITensorNetwork; cutoff, maxdim, normalize=false, ortho=false
)
  return apply(ITensor(o, siteinds(ψ)), ψ; cutoff, maxdim, normalize, ortho)
end

"""
Check env is the correct environment for the subset of vertices of tn
"""
function assert_correct_environment(ψ::ITensorNetwork, env::Vector{ITensor}, verts::Vector)
  outer_verts_indices = flatten([commoninds(ψ, e) for e in boundary_edges(ψ, verts)])
  return issetequal(noncommoninds(env...), outer_verts_indices)
end

### Full Update Routines ###

function create_b(
  p::ITensor,
  q::ITensor,
  gate::ITensor,
  envs::Vector{ITensor},
  bottom_tensor::ITensor;
  opt_sequence=nothing,
)
  return noprime(
    ITensors.contract(
      vcat(ITensor[p, q, gate, dag(prime(bottom_tensor))], envs); sequence=opt_sequence
    ),
  )
end

function M_p(
  envs::Vector{ITensor}, p_q_tensor::ITensor, apply_tensor::ITensor; opt_sequence=nothing
)
  s_ind = setdiff(
    inds(p_q_tensor), collect(Iterators.flatten(inds.(vcat(envs, apply_tensor))))
  )
  return noprime(
    ITensors.contract(
      vcat(
        ITensor[
          p_q_tensor, replaceinds(prime(dag(p_q_tensor)), prime(s_ind), s_ind), apply_tensor
        ],
        envs,
      );
      sequence=opt_sequence,
    ),
  )
end

"""Calculate the overlap of the gate acting on the previous p and q versus the new p and q in the presence of environments. This is the cost function to minimise"""
function fidelity(
  envs::Vector{ITensor},
  p_cur::ITensor,
  q_cur::ITensor,
  p_prev::ITensor,
  q_prev::ITensor,
  gate::ITensor,
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
    envs,
  )
  term1 = ITensors.contract(
    term1_tns; sequence=ITensors.optimal_contraction_sequence(term1_tns)
  )

  term2_tns = vcat(
    [
      p_cur,
      q_cur,
      replaceind(prime(dag(p_cur)), prime(p_sind), p_sind),
      replaceind(prime(dag(q_cur)), prime(q_sind), q_sind),
    ],
    envs,
  )
  term2 = ITensors.contract(
    term2_tns; sequence=ITensors.optimal_contraction_sequence(term2_tns)
  )
  term3_tns = vcat([
    p_prev,
    q_prev,
    prime(dag(p_cur)),
    prime(dag(q_cur)),
    gate,
  ], envs)
  term3 = ITensors.contract(
    term3_tns; sequence=ITensors.optimal_contraction_sequence(term3_tns)
  )

  f = term3[] / sqrt(term1[] * term2[])
  return f * conj(f)
end

"""Do Full Update Sweeping, Optimising the tensors p and q in the presence of the environments envs,
Specifically this functions find the p_cur and q_cur which optimise envs*gate*p*q*dag(prime(p_cur))*dag(prime(q_cur))"""
function optimise_p_q(
  p::ITensor,
  q::ITensor,
  envs::Vector{ITensor},
  o::ITensor;
  nfullupdatesweeps=10,
  maxdim=nothing,
  print_fidelity_loss=false,
  envisposdef=true,
)
  p_cur, q_cur = factorize(apply(o, p * q), inds(p); maxdim, tags=tags(commonind(p, q)))

  fstart = print_fidelity_loss ? fidelity(envs, p_cur, q_cur, p, q, o) : 0

  qs_ind = setdiff(inds(q_cur), collect(Iterators.flatten(inds.(vcat(envs, p_cur)))))
  ps_ind = setdiff(inds(p_cur), collect(Iterators.flatten(inds.(vcat(envs, q_cur)))))

  opt_b_seq = ITensors.optimal_contraction_sequence(
    vcat(ITensor[p, q, o, dag(prime(q_cur))], envs)
  )
  opt_b_tilde_seq = ITensors.optimal_contraction_sequence(
    vcat(ITensor[p, q, o, dag(prime(p_cur))], envs)
  )
  opt_M_seq = ITensors.optimal_contraction_sequence(
    vcat(ITensor[q_cur, replaceinds(prime(dag(q_cur)), prime(qs_ind), qs_ind), p_cur], envs)
  )
  opt_M_tilde_seq = ITensors.optimal_contraction_sequence(
    vcat(ITensor[p_cur, replaceinds(prime(dag(p_cur)), prime(ps_ind), ps_ind), q_cur], envs)
  )
  for i in 1:nfullupdatesweeps
    b = create_b(p, q, o, envs, q_cur; opt_sequence=opt_b_seq)
    M_p_partial = partial(M_p, envs, q_cur; opt_sequence=opt_M_seq)

    p_cur, info = linsolve(M_p_partial, b, p_cur; isposdef=envisposdef, ishermitian=false)

    b_tilde = create_b(p, q, o, envs, p_cur; opt_sequence=opt_b_tilde_seq)
    M_p_tilde_partial = partial(M_p, envs, p_cur; opt_sequence=opt_M_tilde_seq)

    q_cur, info = linsolve(
      M_p_tilde_partial, b_tilde, q_cur; isposdef=envisposdef, ishermitian=false
    )
  end

  fend = print_fidelity_loss ? fidelity(envs, p_cur, q_cur, p, q, o) : 0

  if (print_fidelity_loss && real(fend - fstart) < 0.0 && nsweeps >= 1)
    println(
      "Warning: Krylov Solver Didn't Find a Better Solution by Sweeping. Something might be amiss.",
    )
  end

  return p_cur, q_cur
end
