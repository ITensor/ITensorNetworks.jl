using ITensors, Graphs, NamedGraphs, TimerOutputs
using Distributions, Random
using KaHyPar, Metis
using ITensorNetworks
using ITensorNetworks: ising_network, contract
using OMEinsumContractionOrders

include("exact_contract.jl")
include("3d_cube.jl")
include("sweep_contractor.jl")
include("random_circuit.jl")
include("bench.jl")

Random.seed!(1234)
TimerOutputs.enable_debug_timings(ITensorNetworks)
TimerOutputs.enable_debug_timings(@__MODULE__)
ITensors.set_warn_order(100)

#=
Exact contraction of ising_network
=#
# N = (3, 3, 3)
# beta = 0.3
# network = ising_network(named_grid(N), beta=beta)
# exact_contract(network; sc_target=28)

#=
Exact contraction of randomITensorNetwork
=#
# N = (5, 5, 5)
# distribution = Uniform{Float64}(-0.4, 1.0)
# network = randomITensorNetwork(named_grid(N); link_space=2, distribution=distribution)
# exact_contract(network; sc_target=30) # 47.239753244708396

#=
Bugs
=#
# TODO: (6, 6, 6), env_size=(2, 1, 1) is buggy (cutoff=1e-12, maxdim=256, ansatz="comb", algorithm="density_matrix",)
# TODO below is buggy
# @time bench_3d_cube_lnZ(
#   (3, 8, 10);
#   block_size=(1, 1, 1),
#   beta=0.3,
#   h=0.0,
#   num_iter=2,
#   cutoff=1e-20,
#   maxdim=128,
#   ansatz="mps",
#   algorithm="density_matrix",
#   use_cache=true,
#   ortho=false,
#   env_size=(3, 1, 1),
# )

#=
bench_3d_cube_lnZ
=#
# N = (6, 6, 6)
# beta = 0.3
# network = ising_network(named_grid(N), beta; h=0.0, szverts=nothing)
# tntree = build_tntree(N, network; block_size=(1, 1, 1), env_size=(6, 1, 1))
# @time bench_lnZ(
#   tntree;
#   num_iter=1,
#   cutoff=1e-12,
#   maxdim=128,
#   ansatz="mps",
#   approx_itensornetwork_alg="density_matrix",
#   swap_size=10000,
#   contraction_sequence_alg="sa_bipartite",
#   contraction_sequence_kwargs=(;),
#   linear_ordering_alg="bottom_up",
# )

#=
bench_3d_cube_magnetization
=#
# N = (6, 6, 6)
# beta = 0.22
# h = 0.050
# szverts = [(3, 3, 3)]
# network1 = ising_network(named_grid(N), beta; h=h, szverts=szverts)
# network2 = ising_network(named_grid(N), beta; h=h, szverts=nothing)
# # e1 = exact_contract(network1; sc_target=28)
# # e2 = exact_contract(network2; sc_target=28)
# # @info exp(e1[2] - e2[2])
# tntree1 = build_tntree(N, network1; block_size=(1, 1, 1), env_size=(3, 1, 1))
# tntree2 = build_tntree(N, network2; block_size=(1, 1, 1), env_size=(3, 1, 1))
# @time bench_magnetization(
#   tntree1 => tntree2;
#   num_iter=1,
#   cutoff=1e-12,
#   maxdim=256,
#   ansatz="mps",
#   algorithm="density_matrix",
#   use_cache=true,
#   ortho=false,
#   swap_size=10000,
#   warmup=false,
# )

#=
SweepContractor
=#
# reset_timer!(ITensors.timer)
# # N = (5, 5, 5)
# # beta = 0.3
# # network = ising_network(named_grid(N), beta; h=0.0, szverts=nothing)
# N = (3, 3, 3)
# distribution = Uniform{Float64}(-0.4, 1.0)
# network = randomITensorNetwork(named_grid(N); link_space=2, distribution=distribution)
# ltn = sweep_contractor_tensor_network(
#   network, (i, j, k) -> (j + 0.01 * randn(), k + 0.01 * randn())
# )
# @time lnz = contract_w_sweep(ltn; rank=256)
# @info "lnZ of SweepContractor is", lnz
# show(ITensors.timer)

#=
random regular graph
=#
# nvertices = 220
# deg = 3
# distribution = Uniform{Float64}(-0.2, 1.0)
# network = randomITensorNetwork(
#   distribution, random_regular_graph(nvertices, deg); link_space=2
# )
# # exact_contract(network; sc_target=30)
# # -26.228887728408008 (-1.0, 1.0)
# # 5.633462619348083 (-0.2, 1.0)
# # nvertices_per_partition=10 works 15/20 not work
# # tntree = partitioned_contraction_sequence(network; nvertices_per_partition=10)
# @time bench_lnZ(
#   network;
#   num_iter=2,
#   nvertices_per_partition=10,
#   backend="KaHyPar",
#   cutoff=1e-12,
#   maxdim=512,
#   ansatz="mps",
#   approx_itensornetwork_alg="density_matrix",
#   swap_size=4,
#   contraction_sequence_alg="sa_bipartite",
#   contraction_sequence_kwargs=(;),
#   linear_ordering_alg="bottom_up",
# )

#=
Simulation of random quantum circuit
=#
N = (6, 6)
depth = 6
sequence = random_circuit_line_partition_sequence(N, depth)
@time bench_lnZ(
  sequence;
  num_iter=1,
  cutoff=1e-12,
  maxdim=256,
  ansatz="mps",
  approx_itensornetwork_alg="density_matrix",
  swap_size=8,
  contraction_sequence_alg="sa_bipartite",
  contraction_sequence_kwargs=(;),
  linear_ordering_alg="bottom_up",
)
