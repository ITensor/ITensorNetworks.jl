import ITensorNetworks as itn

function optimal_map(P::itn.ProjTTN, ψ)
  envs = [itn.environment(P, e) for e in itn.incident_edges(P)]
  site_ops = [itn.operator(P)[s] for s in itn.sites(P)]
  contract_list = [envs..., site_ops..., ψ]
  sequence = itn.contraction_sequence(contract_list; alg="optimal")
  Pψ = itn.contract(contract_list; sequence)
  return noprime(Pψ)
end

# This function is a workaround for the slow contraction order
# heuristic in ITensorNetworks/src/treetensornetworks/projttns/projttn.jl
# in the projected_operator_tensors(P::ProjTTN) function (line 97 or so)
function operator_map(P::itn.ProjTTN, ψ)
  ψ = copy(ψ)
  if itn.on_edge(P)
    for edge in itn.incident_edges(P)
      ψ *= itn.environment(P, edge)
    end
  else
    region = itn.sites(P)
    ie = itn.incident_edges(P)
    # TODO: improvement ideas
    # - check which vertex (first(region) vs. last(region)
    #   has more incident edges and contract those environments first
    for edge in ie
      if itn.dst(edge) == first(region)
        ψ *= itn.environment(P, edge)
      end
    end
    for s in itn.sites(P)
      ψ *= itn.operator(P)[s]
    end
    for edge in ie
      if itn.dst(edge) != first(region)
        ψ *= itn.environment(P, edge)
      end
    end
  end
  return noprime(ψ)
end
