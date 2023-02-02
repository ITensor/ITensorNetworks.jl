using ITensors, TimerOutputs, Graphs
using KaHyPar
using ITensorNetworks
using ITensorNetworks: contraction_sequence, ITensorNetwork, ising_network, vertex_tag
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

function exact_contract(N; beta, sc_target)
  ITensors.set_warn_order(1000)
  reset_timer!(timer)
  linkdim = 2
  network = ising_network(named_grid(N), beta)
  tn = Array{ITensor,length(N)}(undef, N...)
  for v in vertices(network)
    tn[v...] = network[v...]
  end
  tn = vec(tn)
  seq = contraction_sequence(tn; alg="kahypar_bipartite", sc_target=sc_target)
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

# if ortho == true
# @info "orthogonalize tn towards the first vertex"
# itn = ITensorNetwork(named_grid(N); link_space=2)
# for i in 1:N[1]
#   for j in 1:N[2]
#     for k in 1:N[3]
#       itn[i, j, k] = tn[i, j, k]
#     end
#   end
# end
# itn = orthogonalize(itn, (1, 1, 1))
# @info itn[1, 1, 1]
# @info itn[1, 1, 1].tensor
# for i in 1:N[1]
#   for j in 1:N[2]
#     for k in 1:N[3]
#       tn[i, j, k] = itn[i, j, k]
#     end
#   end
# end
# end
function build_tntree(N; block_size, beta, h, snake, env_size, szverts)
  @info "beta is", beta
  ITensors.set_warn_order(100)
  network = ising_network(named_grid(N), beta, h; szverts=szverts)
  tn = Array{ITensor,length(N)}(undef, N...)
  for v in vertices(network)
    tn[v...] = network[v...]
  end
  if snake == true
    for k in 1:N[3]
      rangej = iseven(k) ? reverse(1:N[2]) : 1:N[2]
      tn[:, rangej, k] = tn[:, 1:N[2], k]
    end
  end
  if block_size == (1, 1, 1)
    return build_tntree(tn, N; env_size=env_size)
  end
  tn_reduced = ITensorNetwork()
  reduced_N = (
    ceil(Int, N[1] / block_size[1]),
    ceil(Int, N[2] / block_size[2]),
    ceil(Int, N[3] / block_size[3]),
  )
  for i in 1:reduced_N[1]
    for j in 1:reduced_N[2]
      for k in 1:reduced_N[3]
        add_vertex!(tn_reduced, (i, j, k))
        ii = (i - 1) * block_size[1]
        jj = (j - 1) * block_size[2]
        kk = (k - 1) * block_size[3]
        ii_end = min(ii + block_size[1], N[1])
        jj_end = min(jj + block_size[2], N[2])
        kk_end = min(kk + block_size[3], N[3])
        tn_reduced[(i, j, k)] = ITensors.contract(
          tn[(ii + 1):ii_end, (jj + 1):jj_end, (kk + 1):kk_end]...
        )
      end
    end
  end
  for e in edges(tn_reduced)
    v1, v2 = e.src, e.dst
    C = combiner(
      commoninds(tn_reduced[v1], tn_reduced[v2])...;
      tags="$(vertex_tag(v1))↔$(vertex_tag(v2))",
    )
    tn_reduced[v1] = tn_reduced[v1] * C
    tn_reduced[v2] = tn_reduced[v2] * C
  end
  network_reduced = Array{ITensor,3}(undef, reduced_N...)
  for v in vertices(tn_reduced)
    network_reduced[v...] = tn_reduced[v...]
  end
  reduced_env = (
    ceil(Int, env_size[1] / block_size[1]),
    ceil(Int, env_size[2] / block_size[2]),
    ceil(Int, env_size[3] / block_size[3]),
  )
  return build_tntree(network_reduced, reduced_N; env_size=reduced_env)
end

function bench_3d_cube_lnZ(
  N;
  block_size,
  beta,
  h,
  num_iter,
  cutoff,
  maxdim,
  ansatz,
  algorithm,
  snake,
  use_cache,
  ortho,
  env_size,
)
  reset_timer!(timer)
  tntree = build_tntree(
    N;
    block_size=block_size,
    beta=beta,
    h=h,
    snake=snake,
    env_size=env_size,
    szverts=nothing,
  )
  function _run()
    out, log_acc_norm = approximate_contract(
      tntree;
      cutoff=cutoff,
      maxdim=maxdim,
      ansatz=ansatz,
      algorithm=algorithm,
      use_cache=use_cache,
      orthogonalize=ortho,
    )
    @info "out is", log(out[1][1]) + log_acc_norm
    return log(out[1][1]) + log_acc_norm
  end
  out_list = []
  for _ in 1:num_iter
    push!(out_list, _run())
  end
  show(timer)
  # after warmup, start to benchmark
  reset_timer!(timer)
  for _ in 1:num_iter
    push!(out_list, _run())
  end
  @info "lnZ results are", out_list, "mean is", sum(out_list) / (num_iter * 2)
  return show(timer)
end

function bench_3d_cube_magnetization(
  N;
  block_size,
  beta,
  h,
  num_iter,
  cutoff,
  maxdim,
  ansatz,
  algorithm,
  snake,
  use_cache,
  ortho,
  env_size,
  szverts,
)
  reset_timer!(timer)
  tntree1 = build_tntree(
    N;
    block_size=block_size,
    beta=beta,
    h=h,
    snake=snake,
    env_size=env_size,
    szverts=szverts,
  )
  tntree2 = build_tntree(
    N;
    block_size=block_size,
    beta=beta,
    h=h,
    snake=snake,
    env_size=env_size,
    szverts=nothing,
  )
  function _run()
    out, log_acc_norm = approximate_contract(
      tntree1;
      cutoff=cutoff,
      maxdim=maxdim,
      ansatz=ansatz,
      algorithm=algorithm,
      use_cache=use_cache,
      orthogonalize=ortho,
    )
    lognorm1 = log(out[1][1]) + log_acc_norm
    out, log_acc_norm = approximate_contract(
      tntree2;
      cutoff=cutoff,
      maxdim=maxdim,
      ansatz=ansatz,
      algorithm=algorithm,
      use_cache=use_cache,
      orthogonalize=ortho,
    )
    lognorm2 = log(out[1][1]) + log_acc_norm
    return lognorm1 / lognorm2
  end
  out_list = []
  for _ in 1:num_iter
    push!(out_list, _run())
  end
  show(timer)
  # after warmup, start to benchmark
  reset_timer!(timer)
  for _ in 1:num_iter
    push!(out_list, _run())
  end
  @info "magnetization results are", out_list, "mean is", sum(out_list) / (num_iter * 2)
  return show(timer)
end

# exact_contract((4, 4, 10); beta=0.3, sc_target=28)
# TODO: (6, 6, 6), env_size=(2, 1, 1) is buggy (cutoff=1e-12, maxdim=256, ansatz="comb", algorithm="density_matrix",)
# TODO below is buggy
# @time bench_3d_cube_lnZ(
#   (3, 8, 10);
#   use_2D=false,
#   beta=0.3,
#   h=0.0,
#   num_iter=2,
#   cutoff=1e-20,
#   maxdim=128,
#   ansatz="mps",
#   algorithm="density_matrix",
#   snake=false,
#   use_cache=true,
#   ortho=false,
#   env_size=(3, 1, 1),
# )
@time bench_3d_cube_lnZ(
  (6, 6, 6);
  block_size=(6, 1, 1),
  beta=0.3,
  h=0.0,
  num_iter=2,
  cutoff=1e-12,
  maxdim=128,
  ansatz="mps",
  algorithm="density_matrix",
  snake=false,
  use_cache=true,
  ortho=false,
  env_size=(6, 1, 1),
)

# @time bench_3d_cube_magnetization(
#   (1, 6, 6);
#   use_2D=true,
#   beta=0.44,
#   h=0.0001,
#   num_iter=2,
#   cutoff=1e-20,
#   maxdim=64,
#   ansatz="mps",
#   algorithm="density_matrix",
#   snake=false,
#   use_cache=true,
#   ortho=false,
#   env_size=(1, 6, 1),
#   szverts=[(1, 3, 3)],
# )
