using ITensors, TimerOutputs
using ITensorNetworks
using ITensorNetworks: contraction_sequence, ITensorNetwork, ising_network
using ITensorNetworks.ApproximateTNContraction:
  approximate_contract, line_to_tree, timer, line_network

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

function exact_contract(N)
  ITensors.set_warn_order(1000)
  reset_timer!(timer)
  linkdim = 2
  network = ising_network(named_grid(N), 0.3)
  tn = Array{ITensor,length(N)}(undef, N...)
  for v in vertices(network)
    tn[v...] = network[v...]
  end
  tn = vec(tn)
  seq = contraction_sequence(tn; alg="kahypar_bipartite", sc_target=36)
  @info seq
  tn = [(i, 0.0) for i in tn]
  return contract_log_norm(tn, seq)
end

function build_tntree(tn, N; env_size)
  @assert length(N) == length(env_size)
  n = [ceil(Int, N[i] / env_size[i]) for i in 1:length(N)]
  tntree = nothing
  for k in 1:n[3]
    for j in 1:n[2]
      for i in 1:n[1]
        ii = (i - 1) * env_size[1]
        jj = (j - 1) * env_size[2]
        kk = (k - 1) * env_size[3]
        ii_end = min(ii + env_size[1], N[1])
        jj_end = min(jj + env_size[2], N[2])
        kk_end = min(kk + env_size[3], N[3])
        sub_tn = tn[(ii + 1):ii_end, (jj + 1):jj_end, (kk + 1):kk_end]
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

function build_recursive_tntree(tn, N; env_size)
  @assert env_size == (3, 3, 1)
  tn_tree1 = vec(tn[1:3, 1:3, 1])
  tn_tree1 = [vec(tn[1:3, 1:3, 2]), tn_tree1]
  tn_tree1 = [vec(tn[1:3, 1:3, 3]), tn_tree1]

  tn_tree2 = vec(tn[1:3, 4:6, 1])
  tn_tree2 = [vec(tn[1:3, 4:6, 2]), tn_tree2]
  tn_tree2 = [vec(tn[1:3, 4:6, 3]), tn_tree2]

  tn_tree3 = vec(tn[4:6, 1:3, 1])
  tn_tree3 = [vec(tn[4:6, 1:3, 2]), tn_tree3]
  tn_tree3 = [vec(tn[4:6, 1:3, 3]), tn_tree3]

  tn_tree4 = vec(tn[4:6, 4:6, 1])
  tn_tree4 = [vec(tn[4:6, 4:6, 2]), tn_tree4]
  tn_tree4 = [vec(tn[4:6, 4:6, 3]), tn_tree4]

  tn_tree5 = vec(tn[1:3, 1:3, 6])
  tn_tree5 = [vec(tn[1:3, 1:3, 5]), tn_tree5]
  tn_tree5 = [vec(tn[1:3, 1:3, 4]), tn_tree5]

  tn_tree6 = vec(tn[1:3, 4:6, 6])
  tn_tree6 = [vec(tn[1:3, 4:6, 5]), tn_tree6]
  tn_tree6 = [vec(tn[1:3, 4:6, 4]), tn_tree6]

  tn_tree7 = vec(tn[4:6, 1:3, 6])
  tn_tree7 = [vec(tn[4:6, 1:3, 5]), tn_tree7]
  tn_tree7 = [vec(tn[4:6, 1:3, 4]), tn_tree7]

  tn_tree8 = vec(tn[4:6, 4:6, 6])
  tn_tree8 = [vec(tn[4:6, 4:6, 5]), tn_tree8]
  tn_tree8 = [vec(tn[4:6, 4:6, 4]), tn_tree8]
  return [
    [[tn_tree1, tn_tree2], [tn_tree3, tn_tree4]],
    [[tn_tree5, tn_tree6], [tn_tree7, tn_tree8]],
  ]
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
