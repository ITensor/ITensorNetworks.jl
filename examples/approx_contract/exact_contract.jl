using ITensorNetworks: contraction_sequence

INDEX = 0

function contract_log_norm(tn, seq)
  global INDEX
  if seq isa Vector
    if length(seq) == 1
      return seq[1]
    end
    t1 = contract_log_norm(tn, seq[1])
    t2 = contract_log_norm(tn, seq[2])
    @info size(t1[1]), size(t2[1])
    INDEX += 1
    @info "INDEX", INDEX
    out = t1[1] * t2[1]
    nrm = norm(out)
    out /= nrm
    lognrm = log(nrm) + t1[2] + t2[2]
    return (out, lognrm)
  else
    return tn[seq]
  end
end

function exact_contract(network::ITensorNetwork; sc_target=30)
  ITensors.set_warn_order(1000)
  reset_timer!(ITensors.timer)
  tn = Vector{ITensor}(network)
  seq = contraction_sequence(tn; alg="tree_sa")#alg="kahypar_bipartite", sc_target=sc_target)
  @info seq
  tn = [(i, 0.0) for i in tn]
  return contract_log_norm(tn, seq)
end
