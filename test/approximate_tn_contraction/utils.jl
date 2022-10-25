using ITensorNetworkAD.ITensorNetworks:
  ITensor_networks, line_network, TreeTensor, approximate_contract

@profile function peps_contraction_mpomps(tn; cutoff=1e-15, maxdim=1000, snake=false)
  N = size(tn)
  x = tn[:, 1]
  for i in 2:(N[2] - 1)
    A = (iseven(i) && snake) ? reverse(tn[:, i]) : tn[:, i]
    x = contract(MPO(A), MPS(x); cutoff=cutoff, maxdim=maxdim)[:]
  end
  return contract(x..., tn[:, N[2]]...)
end

@profile function contract_w_sweep(tn, rank)
  sweep = sweep_contract(tn, rank, rank)
  return ldexp(sweep...)
end

@profile function contract_element_group(tnet, rank)
  element_grouping = line_network(tnet)
  return approximate_contract(
    element_grouping; cutoff=1e-15, maxdim=rank, maxsize=1e15, algorithm="mincut"
  )
end

@profile function contract_line_group(tnet, rank, N)
  line_grouping = SubNetwork(tnet[:, 1])
  for i in 2:N[2]
    line_grouping = SubNetwork(line_grouping, tnet[:, i]...)
  end
  return batch_tensor_contraction(
    TreeTensor, [line_grouping]; cutoff=1e-15, maxdim=rank, optimize=false
  )
end
