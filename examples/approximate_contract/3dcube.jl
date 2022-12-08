using ITensors, TimerOutputs
using ITensorNetworks.ApproximateTNContraction: approximate_contract
using ITensorNetworks.ApproximateTNContraction: timer, inds_network, ising_partition

function bench_3d_cube(N)
  ITensors.set_warn_order(100)
  reset_timer!(timer)
  linkdim = 2
  maxdim = 1000000
  cutoff = 1e-8
  tn = ising_partition(N, linkdim)
  # tn_inds = inds_network(N...; linkdims=linkdim, periodic=false)
  # tn = map(inds -> randomITensor(inds...), tn_inds)
  # tntree = nothing
  # for k in 1:N[3]
  #   rangej = iseven(k) ? reverse(1:N[2]) : 1:N[2]
  #   for j in rangej
  #     @info j, k
  #     if tntree == nothing
  #       tntree = tn[:, j, k]
  #     else
  #       tntree = [tntree, tn[:, j, k]]
  #     end
  #   end
  # end
  tn = reshape(tn, (N[1], N[2] * N[3]))
  tntree = tn[:, 1]
  for i in 2:(N[2] * N[3])
    tntree = [tntree, tn[:, i]]
  end
  out, log_acc_norm = approximate_contract(
    tntree; cutoff=cutoff, maxdim=maxdim, ansatz="mps", use_cache=true
  )
  @info "out is", log(out[1][1]) + log_acc_norm
  show(timer)
  # after warmup, start to benchmark
  reset_timer!(timer)
  linkdim = 2
  maxdim = 1000000
  cutoff = 1e-8
  tn = ising_partition(N, linkdim)
  tn = reshape(tn, (N[1], N[2] * N[3]))
  tntree = tn[:, 1]
  for i in 2:(N[2] * N[3])
    tntree = [tntree, tn[:, i]]
  end
  out, log_acc_norm = approximate_contract(
    tntree; cutoff=cutoff, maxdim=maxdim, ansatz="mps", use_cache=true
  )
  @info "out is", log(out[1][1]) + log_acc_norm
  return show(timer)
end

bench_3d_cube((5, 5, 5))
