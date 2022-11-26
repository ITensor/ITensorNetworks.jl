using ITensors, TimerOutputs
using ITensorNetworks.ApproximateTNContraction:
  get_tensors,
  OrthogonalITensor,
  tree_approximation_cache,
  inds_binary_tree,
  tree_embedding,
  approximate_contract
using ITensorNetworks.ApproximateTNContraction:
  timer, inds_network, project_boundary, Models, ising_partition

# include("utils.jl")

# @testset "test tree approximation" begin
#   i = Index(2, "i")
#   j = Index(2, "j")
#   k = Index(2, "k")
#   l = Index(2, "l")
#   m = Index(2, "m")
#   n = Index(2, "n")
#   o = Index(2, "o")
#   p = Index(2, "p")
#   q = Index(2, "q")
#   r = Index(2, "r")
#   s = Index(2, "s")
#   t = Index(2, "t")
#   u = Index(2, "u")
#   A = OrthogonalITensor(randomITensor(i, n))
#   B = OrthogonalITensor(randomITensor(j, o))
#   AB = OrthogonalITensor(randomITensor(n, o, r))
#   C = OrthogonalITensor(randomITensor(k, p))
#   D = OrthogonalITensor(randomITensor(l, q))
#   E = OrthogonalITensor(randomITensor(m, u))
#   CD = OrthogonalITensor(randomITensor(p, q, s))
#   ABCD = OrthogonalITensor(randomITensor(r, s, t))
#   ABCDE = OrthogonalITensor(randomITensor(t, u))
#   btree = [[[[i], [j]], [[k], [l]]], [m]]
#   tensors = [A, B, C, D, E, AB, CD, ABCD, ABCDE]
#   embedding = Dict([
#     [i] => [A],
#     [j] => [B],
#     [k] => [C],
#     [l] => [D],
#     [m] => [E],
#     [[i], [j]] => [AB],
#     [[k], [l]] => [CD],
#     [[[i], [j]], [[k], [l]]] => [ABCD],
#     [[[[i], [j]], [[k], [l]]], [m]] => [ABCDE],
#   ])
#   out = tree_approximation_cache(embedding, btree)
#   out = get_tensors(collect(values(out)))
#   @test isapprox(contract(out...), contract(get_tensors(tensors)...))
# end

# @testset "test MPS times MPO" begin
#   N = (5, 3)
#   linkdim = 3
#   cutoff = 1e-15
#   tn_inds = inds_network(N...; linkdims=linkdim)
#   tn = map(inds -> randomITensor(inds...), tn_inds)
#   state = 1
#   tn = project_boundary(tn, state)
#   x, A = tn[:, 1], tn[:, 2]
#   out_true = contract(MPO(A), MPS(x); cutoff=cutoff, maxdim=linkdim * linkdim)
#   out2 = approximate_contract([A, x]; cutoff=cutoff, maxdim=linkdim * linkdim)
#   tsr_true = contract(out_true...)
#   tsr_nrmsquare = (tsr_true * tsr_true)[1]
#   @test isapprox(tsr_true, contract(out2...))

#   maxdims = [2, 4, 6, 8]
#   for dim in maxdims
#     out = contract(MPO(A), MPS(x); cutoff=cutoff, maxdim=dim)
#     out2 = approximate_contract([A, x]; cutoff=cutoff, maxdim=dim)
#     residual1 = tsr_true - contract(out...)
#     residual2 = tsr_true - contract(out2...)
#     error1 = sqrt((residual1 * residual1)[1] / tsr_nrmsquare)
#     error2 = sqrt((residual2 * residual2)[1] / tsr_nrmsquare)
#     print("maxdim, ", dim, ", error1, ", error1, ", error2, ", error2, "\n")
#   end
# end

# @testset "test inds_binary_tree" begin
#   i = Index(2, "i")
#   j = Index(2, "j")
#   k = Index(2, "k")
#   l = Index(2, "l")
#   m = Index(2, "m")
#   n = Index(2, "n")
#   o = Index(2, "o")
#   p = Index(2, "p")

#   T = randomITensor(i, j, k, l, m, n, o, p)
#   M = MPS(T, (i, j, k, l, m, n, o, p); cutoff=1e-5, maxdim=500)
#   network = M[:]

#   out = inds_binary_tree(network, [i, j, k, l, m, n, o, p]; algorithm="mincut")
#   @test length(out) == 2
#   out = inds_binary_tree(network, [i, j, k, l, m, n, o, p]; algorithm="mincut-mps")
#   @test length(out) == 2
#   out = inds_binary_tree(network, [i, j, k, l, m, n, o, p]; algorithm="mps")
#   @test length(out) == 2
# end

# @testset "test inds_binary_tree of a 2D network" begin
#   N = (8, 8, 3)
#   linkdim = 2
#   tn_inds = inds_network(N...; linkdims=linkdim, periodic=false)
#   tn = map(inds -> randomITensor(inds...), tn_inds)
#   network = vec(tn[:, :, 1])
#   out = inds_binary_tree(network, noncommoninds(network...); algorithm="mincut")
#   @test length(out) == 2
#   out = inds_binary_tree(network, noncommoninds(network...); algorithm="mincut-mps")
#   @test length(out) == 2
#   out = inds_binary_tree(network, noncommoninds(network...); algorithm="mps")
#   @test length(out) == 2
# end

# @testset "test tree embedding" begin
#   i = Index(2, "i")
#   j = Index(2, "j")
#   k = Index(2, "k")
#   l = Index(2, "l")
#   m = Index(2, "m")
#   T = randomITensor(i, j, k, l, m)
#   M = MPS(T, (i, j, k, l, m); cutoff=1e-5, maxdim=5)
#   network = M[:]
#   out1 = contract(network...)
#   inds_btree = inds_binary_tree(network, [i, j, k, l, m]; algorithm="mincut")
#   tnet_dict = tree_embedding(network, inds_btree)
#   network2 = vcat(collect(values(tnet_dict))...)
#   out2 = contract(network2...)
#   i1 = noncommoninds(network...)
#   i2 = noncommoninds(network2...)
#   @test (length(i1) == length(i2))
#   @test isapprox(out1, out2)
# end

# function benchmark_peps_contraction(tn; cutoff=1e-15, maxdim=1000, maxsize=10^15)
#   N = size(tn)
#   out = peps_contraction_mpomps(tn; cutoff=cutoff, maxdim=maxdim, snake=false)
#   network = SubNetwork(tn[:, 1])
#   for i in 2:(N[2])
#     network = SubNetwork(network, SubNetwork(tn[:, i]))
#   end
#   out2 = batch_tensor_contraction(
#     TreeTensor, [network]; cutoff=cutoff, maxdim=maxdim, maxsize=maxsize, optimize=false
#   )
#   return out[], ITensor(out2[1])[]
# end

# @testset "test PEPS" begin
#   N = (8, 8) #(12, 12)
#   linkdim = 2
#   cutoff = 1e-15
#   tn_inds = inds_network(N...; linkdims=linkdim, periodic=false)
#   tn = map(inds -> randomITensor(inds...), tn_inds)
#   # tn = ising_partition(N, linkdim)

#   ITensors.set_warn_order(100)
#   maxdim = linkdim^N[2]
#   maxsize = maxdim * maxdim * linkdim
#   out_true, out2 = benchmark_peps_contraction(
#     tn; cutoff=cutoff, maxdim=maxdim, maxsize=maxsize
#   )
#   print(out_true, out2)
#   @test abs((out_true - out2) / out_true) < 1e-3

#   maxdims = [i for i in 2:16]
#   for dim in maxdims
#     size = dim * dim * linkdim
#     out, out2 = benchmark_peps_contraction(tn; cutoff=cutoff, maxdim=dim, maxsize=size)
#     error1 = abs((out - out_true) / out_true)
#     error2 = abs((out2 - out_true) / out_true)
#     print("maxdim, ", dim, ", error1, ", error1, ", error2, ", error2, "\n")
#   end
# end

# function benchmark_3D_contraction(tn; cutoff=1e-15, maxdim=1000, maxsize=10^15)
#   # TODO: this sequential MPS doesn't give the desired tree when each tn[i] is a slice of the 2D surface
#   out = contract(
#     tn[1]; cutoff=cutoff, maxdim=maxdim, maxsize=maxsize, algorithm="sequential-mps"
#   )
#   for i in 2:length(tn)
#     out = contract(
#       out, tn[i]; cutoff=cutoff, maxdim=maxdim, maxsize=maxsize, algorithm="sequential-mps"
#     )
#   end
#   out2 = contract(tn[1]; cutoff=cutoff, maxdim=maxdim, maxsize=maxsize, algorithm="mps")
#   for i in 2:length(tn)
#     out2 = contract(
#       out2, tn[i]; cutoff=cutoff, maxdim=maxdim, maxsize=maxsize, algorithm="mps"
#     )
#   end
#   return out[], out2[]
# end

# @testset "test 3-D cube with 2D grouping" begin
#   reset_timer!(timer)
#   N = (3, 3, 4) #(12, 12)
#   linkdim = 2
#   nrows = prod([s for s in N[1:(length(N) - 1)]])
#   ncols = N[length(N)]
#   maxdim = linkdim^(floor(nrows))
#   cutoff = 1e-15
#   # tn = ising_partition(N, linkdim)
#   tn_inds = inds_network(N...; linkdims=linkdim, periodic=false)
#   tn = map(inds -> randomITensor(inds...), tn_inds)
#   # snake mapping
#   for k in 1:N[3]
#     for j in 1:N[2]
#       if iseven(j)
#         tn[:, j, k] = reverse(tn[:, j, k])
#       end
#     end
#   end
#   tn = reshape(tn, (nrows, ncols))
#   tn = [TreeTensor(tn[:, i]) for i in 1:ncols]
#   @info size(tn)
#   ITensors.set_warn_order(100)
#   maxsize = maxdim * maxdim * linkdim
#   out1, out2 = benchmark_3D_contraction(tn; cutoff=cutoff, maxdim=maxdim, maxsize=maxsize)
#   show(timer)
#   print(out1, out2)
#   @test abs((out1 - out2) / out1) < 1e-3
#   maxdims = [3, 5, 8, 10, 11, 12, 13, 14, 15, 16, 20, 31, 32]
#   for dim in maxdims
#     size = dim * dim * linkdim
#     out, out2 = benchmark_3D_contraction(tn; cutoff=cutoff, maxdim=dim, maxsize=size)
#     error1 = abs((out - out1) / out1)
#     error2 = abs((out2 - out1) / out1)
#     print("maxdim, ", dim, ", error1, ", error1, ", error2, ", error2, "\n")
#   end
# end

@testset "test 3-D cube with 1D grouping" begin
  ITensors.set_warn_order(100)
  reset_timer!(timer)
  N = (5, 5, 5)
  linkdim = 2
  maxdim = 5
  cutoff = 1e-15
  # tn = ising_partition(N, linkdim)
  tn_inds = inds_network(N...; linkdims=linkdim, periodic=false)
  tn = map(inds -> randomITensor(inds...), tn_inds)
  tn = reshape(tn, (N[1], N[2] * N[3]))
  tntree = tn[:, 1]
  for i in 2:(N[2] * N[3])
    tntree = [tntree, tn[:, i]]
  end
  out = approximate_contract(tntree; cutoff=cutoff, maxdim=maxdim)
  @info "out is", out[1][1]
  show(timer)
  # after warmup, start to benchmark
  reset_timer!(timer)
  N = (5, 5, 5)
  linkdim = 2
  maxdim = 5
  cutoff = 1e-15
  tn = ising_partition(N, linkdim)
  tn = reshape(tn, (N[1], N[2] * N[3]))
  tntree = tn[:, 1]
  for i in 2:(N[2] * N[3])
    tntree = [tntree, tn[:, i]]
  end
  out = approximate_contract(tntree; cutoff=cutoff, maxdim=maxdim)
  @info "out is", out[1][1]
  show(timer)
end

# #TODO
# # @testset "test 3-D cube with DMRG-like algorithm" begin
# #   ITensors.set_warn_order(100)
# #   do_profile(true)
# #   N = (5, 5, 3) # (5, 5, 5)
# #   linkdim = 2
# #   maxdim = linkdim^(floor(N[1] * N[2]))
# #   cutoff = 1e-15
# #   tn = ising_partition(N, linkdim)
# #   function build_tree(i)
# #     tree = tn[:, 1, i]
# #     for j in 2:N[2]
# #       tree = [tree, tn[:, j, i]]
# #     end
# #     return tree
# #   end
# #   tn1, _ = approximate_contract(
# #     build_tree(1); cutoff=cutoff, maxdim=maxdim, maxsize=1e15, algorithm="mps"
# #   )
# #   tn2, _ = approximate_contract(
# #     build_tree(2); cutoff=cutoff, maxdim=maxdim, maxsize=1e15, algorithm="mps"
# #   )
# #   tn3, _ = approximate_contract(
# #     build_tree(3); cutoff=cutoff, maxdim=maxdim, maxsize=1e15, algorithm="mps"
# #   )
# #   tn12, _ = approximate_contract(
# #     [tn1..., tn2...]; cutoff=cutoff, maxdim=maxdim, maxsize=1e15, algorithm="mps"
# #   )
# #   out, _ = approximate_contract(
# #     [tn12..., tn3...]; cutoff=cutoff, maxdim=maxdim, maxsize=1e15, algorithm="mps"
# #   )
# # profile_exit()
# # end

# @testset "benchmark PEPS" begin
#   N = (8, 8) #(12, 12)
#   linkdim = 10
#   cutoff = 1e-15
#   tn_inds = inds_network(N...; linkdims=linkdim, periodic=false)

#   dim = 20
#   size = dim * dim * linkdim
#   # warmup
#   for i in 1:2
#     tn = map(inds -> randomITensor(inds...), tn_inds)
#     ITensors.set_warn_order(100)
#     benchmark_peps_contraction(tn; cutoff=cutoff, maxdim=dim, maxsize=size)
#   end

#   do_profile(true)
#   for i in 1:3
#     tn = map(inds -> randomITensor(inds...), tn_inds)
#     ITensors.set_warn_order(100)
#     benchmark_peps_contraction(tn; cutoff=cutoff, maxdim=dim, maxsize=size)
#   end
#   profile_exit()
# end
