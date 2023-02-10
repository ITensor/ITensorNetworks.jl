
"""
Check env is the correct environment for the subset of vertices of tn
"""
function assert_correct_environment(ψ::ITensorNetwork, env::Vector{ITensor}, verts::Vector)
  outer_verts_indices = flatten([commoninds(ψ, e) for e in boundary_edges(ψ, verts)])
  return issetequal(noncommoninds(env...), outer_verts_indices)
end

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
  s_ind = setdiff(inds(p_q_tensor), collect(Iterators.flatten(inds.(vcat(envs, apply_tensor)))))
  return noprime(
    ITensors.contract(
      vcat(
        ITensor[p_q_tensor, replaceinds(prime(dag(p_q_tensor)), prime(s_ind), s_ind), apply_tensor],
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

  p_sind = commonind(p_cur, gate)
  q_sind = commonind(q_cur, gate)
  gate_sq = gate*prime(dag(gate))
  replaceinds!(gate_sq, noncommoninds(gate_sq, gate), Index[prime(p_sind), prime(q_sind)])
  term1_tns = vcat(
    [
      p_prev,
      q_prev,
      prime(dag(p_prev)),
      prime(dag(q_prev)),
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
  term3_tns = vcat(
    [

      p_prev,
      q_prev,
      prime(dag(p_cur)),
      prime(dag(q_cur)),
      gate
    ],
    envs,
  )
  term3 = ITensors.contract(
    term3_tns; sequence=ITensors.optimal_contraction_sequence(term3_tns)
  )

  f = term3[] / sqrt(term1[] * term2[])
  return f * conj(f)
end

"""Do Full Update Sweeping, Optimising the tensors p and q in the presence of the environments envs"""
function optimise_p_q(
  p::ITensor,
  q::ITensor,
  envs::Vector{ITensor},
  gate::ITensor,
  nsweeps::Int64;
  maxdim=nothing,
  print_fidelity_loss=false,
  isposdef=true,
)
  p_cur, q_cur = factorize(noprime(p * q * gate), inds(p); maxdim, tags = tags(commonind(p, q)))
  normalize!(p_cur)
  normalize!(q_cur)

  fstart = print_fidelity_loss ? fidelity(envs, p_cur, q_cur, p, q, gate) : 0

  #Get 'site'-style ind on p and q (useful)
  qs_ind = setdiff(inds(q_cur), collect(Iterators.flatten(inds.(vcat(envs, p_cur)))))
  ps_ind = setdiff(inds(p_cur), collect(Iterators.flatten(inds.(vcat(envs, q_cur)))))

  opt_b_seq = ITensors.optimal_contraction_sequence(
    vcat(ITensor[p, q, gate, dag(prime(q_cur))], envs)
  )
  opt_b_tilde_seq = ITensors.optimal_contraction_sequence(
    vcat(ITensor[p, q, gate, dag(prime(p_cur))], envs)
  )
  opt_M_seq = ITensors.optimal_contraction_sequence(
    vcat(ITensor[q_cur, replaceinds(prime(dag(q_cur)), prime(qs_ind), qs_ind), p_cur], envs)
  )
  opt_M_tilde_seq = ITensors.optimal_contraction_sequence(
    vcat(ITensor[p_cur, replaceinds(prime(dag(p_cur)), prime(ps_ind), ps_ind), q_cur], envs)
  )
  for i in 1:nsweeps
    b = create_b(p, q, gate, envs, q_cur; opt_sequence=opt_b_seq)
    M_p_partial = partial(M_p, envs, q_cur; opt_sequence=opt_M_seq)

    p_cur, info = linsolve(
      M_p_partial,
      b,
      p_cur;
      isposdef=isposdef,
      ishermitian=false,
      orth=KrylovKit.ModifiedGramSchmidt(),
    )
    normalize!(p_cur)

    b_tilde = create_b(p, q, gate, envs, p_cur; opt_sequence=opt_b_tilde_seq)
    M_p_tilde_partial = partial(M_p, envs, p_cur; opt_sequence=opt_M_tilde_seq)

    q_cur, info = linsolve(
      M_p_tilde_partial,
      b_tilde,
      q_cur;
      isposdef=isposdef,
      ishermitian=false,
      orth=KrylovKit.ModifiedGramSchmidt(),
    )
    normalize!(q_cur)
  end

  fend = print_fidelity_loss ? fidelity(envs, p_cur, q_cur, p, q, gate) : 0

  if (print_fidelity_loss && real(fend - fstart) < 0.0 && nsweeps >= 1)
    println(
      "Warning: Krylov Solver Didn't Find a Better Solution by Sweeping. Something might be amiss.",
    )
  end

  return p_cur, q_cur
end

"""Do a full update on the ITensorNetwork ψ with the gate o and in the presence of a series of environment tensors
Note the environment tensors should be the environments for <ψ|ψ> (i.e. the braket, not just the bra)"""
function apply_fullupdate(
  o::ITensor,
  ψ::AbstractITensorNetwork,
  envs::Vector{ITensor};
  nsweeps=10,
  maxdim=nothing,
  normalize=false,
  print_fidelity_loss=false,
  isposdef=true,
)
  ψ = copy(ψ)
  v⃗ = neighbor_vertices(ψ, o)
  if length(v⃗) == 1 || isempty(envs)
    return ITensors.apply(o, ψ; maxdim, normalize, ortho)
  elseif length(v⃗) == 2
    e = v⃗[1] => v⃗[2]
    if !has_edge(ψ, e)
      error("Vertices where the gates are being applied must be neighbors for now.")
    end

    if (
      !assert_correct_environment(
        ψ ⊗ prime(dag(ψ); sites=[]), envs, [(v⃗[1], 1), (v⃗[1], 2), (v⃗[2], 1), (v⃗[2], 2)]
      )
    )
      println(
        "Error: Environment provided does not match the local wavefunction, cannot apply"
      )
      return tn
    end

    X, p = factorize(
      ψ[v⃗[1]], uniqueinds(uniqueinds(ψ[v⃗[1]], ψ[v⃗[2]]), inds(ψ[v⃗[1]]; tags="Site"))
    )
    Y, q = factorize(
      ψ[v⃗[2]], uniqueinds(uniqueinds(ψ[v⃗[2]], ψ[v⃗[1]]), inds(ψ[v⃗[2]]; tags="Site"))
    )

    envs = copy(envs)
    envs = vcat(envs, X, prime(dag(X)), Y, prime(dag(Y)))

    p, q = optimise_p_q(p, q, envs, o, nsweeps; maxdim, print_fidelity_loss, isposdef)

    ψᵥ₁, ψᵥ₂ = X * p, Y * q

    if normalize
      ψᵥ₁ ./= norm(ψᵥ₁)
      ψᵥ₂ ./= norm(ψᵥ₂)
    end

    ψ[v⃗[1]] = ψᵥ₁
    ψ[v⃗[2]] = ψᵥ₂

    return ψ

  elseif length(v⃗) < 1
    error("Gate being applied does not share indices with tensor network.")
  elseif length(v⃗) > 2
    error("Gates with more than 2 sites is not supported yet.")
  end
end

partial = (f, a...; c...) -> (b...) -> f(a..., b...; c...)
