using ITensorNetworks: vertex_tag

function build_tntree(tn, N; env_size)
  @assert length(N) == length(env_size)
  n = [ceil(Int, N[i] / env_size[i]) for i in 1:length(N)]
  tntree = nothing
  ranges = [1:i for i in n]
  for coord in Iterators.product(ranges...)
    @info coord
    block_coord_floor = [(i - 1) * s for (i, s) in zip(coord, env_size)]
    block_coord_ceil = [min(s + f, n) for (s, f, n) in zip(env_size, block_coord_floor, N)]
    sub_tn = tn[collect(
      (f + 1):c for (f, c) in zip(block_coord_floor, block_coord_ceil)
    )...]
    sub_tn = vec(sub_tn)
    if tntree == nothing
      tntree = sub_tn
    else
      tntree = [tntree, sub_tn]
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
function build_tntree(N, network::ITensorNetwork; block_size::Tuple, env_size::Tuple)
  @assert length(block_size) == length(env_size)
  order = length(block_size)
  tn = Array{ITensor,length(N)}(undef, N...)
  for v in vertices(network)
    tn[v...] = network[v...]
  end
  if block_size == Tuple(1 for _ in 1:order)
    return build_tntree(tn, N; env_size=env_size)
  end
  tn_reduced = ITensorNetwork()
  reduced_N = [ceil(Int, x / y) for (x, y) in zip(N, block_size)]
  ranges = [1:n for n in reduced_N]
  for coord in Iterators.product(ranges...)
    add_vertex!(tn_reduced, coord)
    block_coord_floor = [(i - 1) * s for (i, s) in zip(coord, block_size)]
    block_coord_ceil = [
      min(s + f, n) for (s, f, n) in zip(block_size, block_coord_floor, N)
    ]
    tn_reduced[coord] = ITensors.contract(
      tn[collect((f + 1):c for (f, c) in zip(block_coord_floor, block_coord_ceil))...]...
    )
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
  network_reduced = Array{ITensor,length(reduced_N)}(undef, reduced_N...)
  for v in vertices(tn_reduced)
    network_reduced[v...] = tn_reduced[v...]
  end
  reduced_env = [ceil(Int, x / y) for (x, y) in zip(env_size, block_size)]
  return build_tntree(network_reduced, reduced_N; env_size=reduced_env)
end
