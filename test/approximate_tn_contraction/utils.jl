using TimerOutputs
using ITensorNetworks.ApproximateTNContraction: timer, line_network, approximate_contract

function peps_contraction_mpomps(tn; cutoff=1e-15, maxdim=1000, snake=false)
  @timeit timer "peps_contraction_mpomps" begin
    N = size(tn)
    x = tn[:, 1]
    for i in 2:(N[2] - 1)
      A = (iseven(i) && snake) ? reverse(tn[:, i]) : tn[:, i]
      x = contract(MPO(A), MPS(x); cutoff=cutoff, maxdim=maxdim)[:]
    end
    return contract(x..., tn[:, N[2]]...)
  end
end

function contract_w_sweep(tn, rank)
  @timeit timer "contract_w_sweep" begin
    sweep = sweep_contract(tn, rank, rank)
    return ldexp(sweep...)
  end
end

function contract_element_group(tnet; cutoff, maxdim)
  @timeit timer "contract_element_group" begin
    element_grouping = line_network(tnet)
    return approximate_contract(element_grouping; cutoff=cutoff, maxdim=maxdim)
  end
end

function contract_line_group(tnet; cutoff, maxdim)
  N = size(tnet)
  @timeit timer "contract_line_group" begin
    tntree = tnet[:, 1]
    for i in 2:N[2]
      tntree = [tntree, tnet[:, i]]
    end
    return approximate_contract(tntree; cutoff=cutoff, maxdim=maxdim)
  end
end
