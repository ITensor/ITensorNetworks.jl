using ITensors, TimerOutputs
using ITensorNetworks
using ITensorNetworks: contraction_sequence, ITensorNetwork, ising_network
using ITensorNetworks.ApproximateTNContraction:
  approximate_contract, line_to_tree, timer, line_network

function contract_log_norm(tn, seq)
  if seq isa Vector
    t1 = contract_log_norm(tn, seq[1])
    t2 = contract_log_norm(tn, seq[2])
    @info size(t1[1]), size(t2[1])
    out = t1[1] * t2[1]
    nrm = norm(out)
    out /= nrm
    lognrm = log(nrm) + t1[2] + t2[2]
    return (out, lognrm)
  else
    return tn[seq]
  end
end

function exact_contract(N)
  ITensors.set_warn_order(100)
  reset_timer!(timer)
  linkdim = 2
  tn = vec(ising_partition(N, linkdim))
  # contraction_sequence(tn; alg="kahypar_bipartite", sc_target=30)
  seq = line_to_tree([i for i in 1:prod(N)])
  tn = [(i, 0.0) for i in tn]
  return contract_log_norm(tn, seq)
end

function build_tntree(tn, N; env_size)
  @assert length(N) == length(env_size)
  n = [Integer(N[i] / env_size[i]) for i in 1:length(N)]
  tntree = nothing
  for k in 1:n[3]
    for j in 1:n[2]
      for i in 1:n[1]
        ii = (i - 1) * env_size[1]
        jj = (j - 1) * env_size[2]
        kk = (k - 1) * env_size[3]
        sub_tn = tn[
          (ii + 1):(ii + env_size[1]),
          (jj + 1):(jj + env_size[2]),
          (kk + 1):(kk + env_size[3]),
        ]
        sub_tn = vec(sub_tn)
        if tntree == nothing
          tntree = sub_tn
        else
          tntree = [tntree, sub_tn]
        end
      end
    end
  end
  return tntree
end

function bench_3d_cube(
  N; num_iter, cutoff, maxdim, ansatz, snake, use_cache, ortho, env_size
)
  ITensors.set_warn_order(100)
  reset_timer!(timer)
  network = ising_network(named_grid(N), 0.3)
  tn = Array{ITensor,length(N)}(undef, N...)
  for v in vertices(network)
    tn[v...] = network[v...]
  end
  # if ortho == true
  #   @info "orthogonalize tn towards the first vertex"
  #   itn = ITensorNetwork(named_grid(N); link_space=2)
  #   for i in 1:N[1]
  #     for j in 1:N[2]
  #       for k in 1:N[3]
  #         itn[i, j, k] = tn[i, j, k]
  #       end
  #     end
  #   end
  #   itn = orthogonalize(itn, (1, 1, 1))
  #   @info itn[1, 1, 1]
  #   @info itn[1, 1, 1].tensor
  #   for i in 1:N[1]
  #     for j in 1:N[2]
  #       for k in 1:N[3]
  #         tn[i, j, k] = itn[i, j, k]
  #       end
  #     end
  #   end
  # end
  if snake == true
    for k in 1:N[3]
      rangej = iseven(k) ? reverse(1:N[2]) : 1:N[2]
      tn[:, rangej, k] = tn[:, 1:N[2], k]
    end
  end
  tntree = build_tntree(tn, N; env_size=env_size)
  out_list = []
  for _ in 1:num_iter
    out, log_acc_norm = approximate_contract(
      tntree;
      cutoff=cutoff,
      maxdim=maxdim,
      ansatz=ansatz,
      use_cache=use_cache,
      orthogonalize=ortho,
    )
    @info "out is", log(out[1][1]) + log_acc_norm
    push!(out_list, log(out[1][1]) + log_acc_norm)
  end
  show(timer)
  # after warmup, start to benchmark
  reset_timer!(timer)
  for _ in 1:num_iter
    out, log_acc_norm = approximate_contract(
      tntree;
      cutoff=cutoff,
      maxdim=maxdim,
      ansatz=ansatz,
      use_cache=use_cache,
      orthogonalize=ortho,
    )
    @info "out is", log(out[1][1]) + log_acc_norm
    push!(out_list, log(out[1][1]) + log_acc_norm)
  end
  @info "lnZ results are", out_list, "mean is", sum(out_list) / (num_iter * 2)
  return show(timer)
end

# exact_contract((5, 5, 5))
bench_3d_cube(
  (6, 6, 6);
  num_iter=2,
  cutoff=1e-8,
  maxdim=32,
  ansatz="mps",
  snake=false,
  use_cache=true,
  ortho=false,
  env_size=(2, 2, 1),
)
