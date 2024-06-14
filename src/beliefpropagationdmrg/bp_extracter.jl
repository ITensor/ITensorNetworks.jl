using ITensors: scalartype, which_op, name, names, sites
using ITensorNetworks:
  ket_vertices, bra_vertices, tensornetwork, default_message_update, operator_network
using ITensorNetworks.ITensorsExtensions: map_eigvals
using NamedGraphs.GraphsExtensions: a_star, neighbors

function effective_environments(state::ITensorNetwork, H::OpSum, ψIψ_bpc::BeliefPropagationCache, region)
  environments = Vector{ITensor}[]
  s = siteinds(state)
  op_tensors = Vector{ITensor}(H, s)
  for (i, term) in enumerate(H)
    term_envs = ITensor[]
    if length(sites(term)) == 1 && !iszero(first(term.args))
      path = a_star(state, only(sites(term)), only(region))
      ψOψ_qf = QuadraticFormNetwork(state)
      ψOψ_qf[(only(sites(term)), "operator")] = op_tensors[i]
    elseif length(sites(term)) == 2 && !iszero(first(term.args))
      v1, v2 = first(sites(term)), last(sites(term))
      path = a_star(state, v1, only(region))
      if v2 ∉ src.(path) && v2 ∉ dst.(path)
        prepend!(path, NamedEdge[NamedEdge(v2 => v1)])
      end
      Ov1, Ov2 = factorize_svd(op_tensors[i], s[v1], s[v1]'; cutoff = 1e-16)
      ψOψ_qf = QuadraticFormNetwork(state)
      ψOψ_qf[(v1, "operator")] = Ov1
      ψOψ_qf[(v2, "operator")] = Ov2
    else 
      path = nothing
    end
    if !isnothing(path)
      env = ITensor(1.0)
      path = vcat(src.(path))
      for v in path
        env *= ψOψ_qf[(v, "bra")]* ψOψ_qf[(v, "operator")] * ψOψ_qf[(v, "ket")]
        vns = neighbors(state, v)
        for vn in vns
          if vn ∉ path && vn != only(region)
            env *= only(message(ψIψ_bpc, PartitionEdge(vn => v)))
          end
        end
      end
      push!(term_envs, env)
      for vn in neighbors(state, only(region))
        if vn ∉ path
          push!(term_envs, only(message(ψIψ_bpc, PartitionEdge(vn => only(region)))))
        end
      end
      push!(term_envs, ψOψ_qf[(only(region), "operator")])

      push!(environments, term_envs)
    end
  end

  return environments
end

function bp_extracter(
  ψ::AbstractITensorNetwork,
  H::OpSum,
  ψIψ_bpc::BeliefPropagationCache,
  region;
  regularization=10 * eps(scalartype(ψ)),
  ishermitian=true,
)

  form_network = tensornetwork(ψIψ_bpc)
  form_ket_vertices, form_bra_vertices = ket_vertices(form_network, region),
  bra_vertices(form_network, region)

  ∂ψOψ_bpc_∂rs = effective_environments(ψ, H, ψIψ_bpc, region)
  state = prod([ψ[v] for v in region])
  messages = environment(ψIψ_bpc, partitionvertices(ψIψ_bpc, form_ket_vertices))
  f_sqrt = sqrt ∘ (x -> x + regularization)
  f_inv_sqrt = inv ∘ sqrt ∘ (x -> x + regularization)
  sqrt_mts =
    map_eigvals.(
      (f_sqrt,), messages, first.(inds.(messages)), last.(inds.(messages)); ishermitian
    )
  inv_sqrt_mts =
    map_eigvals.(
      (f_inv_sqrt,), messages, first.(inds.(messages)), last.(inds.(messages)); ishermitian
    )

  return state, ∂ψOψ_bpc_∂rs, sqrt_mts, inv_sqrt_mts
end