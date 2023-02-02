using Graphs
using NamedGraphs
using ITensors
using ITensorNetworks
using ITensorNetworks: contract_inner, assert_correct_environment, get_environment
using Compat
using KaHyPar
using LinearAlgebra
using KrylovKit
using Random

"""
Check env is the correct environment for the subset of vertices of tn
"""
function assert_correct_environment(tn::ITensorNetwork, env::Vector{ITensor}, verts::Vector)
  outer_verts_indices = flatten([commoninds(tn, e) for e in boundary_edges(tn, verts)])
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

function initial_guess_pprime_qprime(
  p::ITensor, q::ITensor, gate::ITensor; cutoff=nothing, maxdim=nothing
)
  p_out, q_out = factorize(noprime(p * q * gate), inds(p); cutoff, maxdim)

  cur_ind = commonind(p_out, q_out)
  replaceind!(p_out, cur_ind, replacetags(cur_ind, tags(cur_ind), tags(commonind(p, q))))
  replaceind!(q_out, cur_ind, replacetags(cur_ind, tags(cur_ind), tags(commonind(p, q))))

  return p_out, q_out
end

function M_p(
  envs::Vector{ITensor}, p_q_tensor::ITensor, apply_tensor::ITensor; opt_sequence=nothing
)
  return noprime(
    ITensors.contract(
      vcat(
        ITensor[p_q_tensor, noprime!(prime(dag(p_q_tensor)); tags="Site"), apply_tensor],
        envs,
      );
      sequence=opt_sequence,
    ),
  )
end

function fidelity(
  envs::Vector{ITensor},
  p_cur::ITensor,
  q_cur::ITensor,
  p_prev::ITensor,
  q_prev::ITensor,
  gate::ITensor,
)
  t1_tns = vcat(
    [
      p_prev,
      q_prev,
      prime(dag(p_prev)),
      prime(dag(q_prev)),
      swapprime(gate * swapprime(dag(gate), 0, 2), 2, 1),
    ],
    envs,
  )
  t1 = ITensors.contract(t1_tns; sequence=ITensors.optimal_contraction_sequence(t1_tns))
  t2_tns = vcat(
    [
      p_cur,
      q_cur,
      noprime(prime(dag(p_cur)); tags="Site"),
      noprime(prime(dag(q_cur)); tags="Site"),
    ],
    envs,
  )
  t2 = ITensors.contract(t2_tns; sequence=ITensors.optimal_contraction_sequence(t2_tns))
  t3_tns = vcat(
    [
      replaceprime(dag(prime(p_prev * q_prev * gate)), 2 => 1),
      prime(p_cur * q_cur; tags="Site"),
    ],
    envs,
  )
  t3 = ITensors.contract(t3_tns; sequence=ITensors.optimal_contraction_sequence(t3_tns))

  f = t3[] / sqrt(t1[] * t2[])
  return f * conj(f)
end

function optimise_p_q(
  X::ITensor,
  Y::ITensor,
  p::ITensor,
  q::ITensor,
  envs::Vector{ITensor},
  gate::ITensor,
  nsweeps::Int64;
  cutoff=nothing,
  maxdim=nothing,
  fidelity_loss=false,
  isposdef=true,
)
  p_cur, q_cur = initial_guess_pprime_qprime(p, q, gate; cutoff, maxdim)
  normalize!(p_cur)
  normalize!(q_cur)

  env_XY = vcat(envs, X, prime(dag(X)), Y, prime(dag(Y)))

  fstart = fidelity_loss ? fidelity(env_XY, p_cur, q_cur, p, q, gate) : 0

  opt_b_seq = ITensors.optimal_contraction_sequence(
    vcat(ITensor[p, q, gate, dag(prime(q_cur))], env_XY)
  )
  opt_b_tilde_seq = ITensors.optimal_contraction_sequence(
    vcat(ITensor[p, q, gate, dag(prime(p_cur))], env_XY)
  )
  opt_M_seq = ITensors.optimal_contraction_sequence(
    vcat(ITensor[q_cur, noprime!(prime(dag(q_cur)); tags="Site"), p_cur], env_XY)
  )
  opt_M_tilde_seq = ITensors.optimal_contraction_sequence(
    vcat(ITensor[p_cur, noprime!(prime(dag(p_cur)); tags="Site"), q_cur], env_XY)
  )
  for i in 1:nsweeps
    b = create_b(p, q, gate, env_XY, q_cur; opt_sequence=opt_b_seq)
    M_p_partial = partial(M_p, env_XY, q_cur; opt_sequence=opt_M_seq)

    p_cur, info = linsolve(
      M_p_partial,
      b,
      p_cur;
      isposdef=isposdef,
      ishermitian=false,
      orth=KrylovKit.ModifiedGramSchmidt(),
    )
    normalize!(p_cur)

    b_tilde = create_b(p, q, gate, env_XY, p_cur; opt_sequence=opt_b_tilde_seq)
    M_p_tilde_partial = partial(M_p, env_XY, p_cur; opt_sequence=opt_M_tilde_seq)

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

  fend = fidelity_loss ? fidelity(env_XY, p_cur, q_cur, p, q, gate) : 0

  if (fidelity_loss && real(fend - fstart) < 0.0 && nsweeps >= 1)
    println("Warning Krylov Solver Didn't Find a Better Solution by Sweeping")
  end

  return p_cur, q_cur, fend
end

function apply_gate(
  tn::ITensorNetwork,
  gate::ITensor,
  mts::DataGraph,
  v1,
  v2;
  nsweeps=5,
  maxdim=nothing,
  fidelity_loss=false,
)
  envs = get_environment(tn, mts, [(v1, 1), (v1, 2), (v2, 1), (v2, 2)])
  if (!assert_correct_environment(tn, envs, [(v1, 1), (v1, 2), (v2, 1), (v2, 2)]))
    println(
      "ERROR: ENVIRONMENT DOES NOT MATCH THE LOCAL WAVEFUNCTION, CANNOT APPLY THE GATE"
    )
    return tn
  end

  X, p = factorize(
    tn[(v1, 1)],
    uniqueinds(uniqueinds(tn[(v1, 1)], tn[(v2, 1)]), inds(tn[(v1, 1)]; tags="Site")),
  )
  Y, q = factorize(
    tn[(v2, 1)],
    uniqueinds(uniqueinds(tn[(v2, 1)], tn[(v1, 1)]), inds(tn[(v2, 1)]; tags="Site")),
  )

  p, q, f = optimise_p_q(
    X, Y, p, q, envs, gate, nsweeps; maxdim=maxdim, fidelity_loss=fidelity_loss
  )

  ψv1, ψv2 = X * p, Y * q

  return ψv1 / norm(ψv1), ψv2 / norm(ψv2), f
end

partial = (f, a...; c...) -> (b...) -> f(a..., b...; c...)
