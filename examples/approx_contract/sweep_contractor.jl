using Distributions, Random, TimerOutputs
using SweepContractor
using NamedGraphs, ITensors, ITensorNetworks
using ITensorNetworks: ising_network

"""
Construct `SweepContractor.LabelledTensorNetwork` based on the input `tn` and the input
`position_func`. `position_func` maps each vertex in `tn` to a 2D coordimate (x, y).
"""
function sweep_contractor_tensor_network(tn::ITensorNetwork, position_func::Function)
  ltn = SweepContractor.LabelledTensorNetwork{vertextype(tn)}()
  for v in vertices(tn)
    neighbor_vs = neighbors(tn, v)
    adj = Vector{vertextype(tn)}()
    for ind in inds(tn[v])
      for neighbor_v in neighbor_vs
        if ind in commoninds(tn[v], tn[neighbor_v])
          push!(adj, neighbor_v)
        end
      end
    end
    @assert setdiff(adj, neighbor_vs) == []
    arr = tn[v].tensor.storage.data
    arr = reshape(arr, [ITensors.dim(i) for i in inds(tn[v])]...)
    ltn[v...] = SweepContractor.Tensor(adj, arr, position_func(v...)...)
  end
  return ltn
end

function contract_w_sweep(ltn::SweepContractor.LabelledTensorNetwork; rank)
  @timeit_debug ITensors.timer "contract with SweepContractor" begin
    sweep = sweep_contract(ltn, rank, rank)
    return log(abs(sweep[1])) + sweep[2] * log(2)
  end
end

Random.seed!(1234)
TimerOutputs.enable_debug_timings(@__MODULE__)
reset_timer!(ITensors.timer)

N = (3, 3, 3)
beta = 0.3
network = ising_network(named_grid(N), beta; h=0.0, szverts=nothing)
# N = (5, 5, 5)
# distribution = Uniform{Float64}(-0.4, 1.0)
# network = randomITensorNetwork(named_grid(N); link_space=2, distribution=distribution)
ltn = sweep_contractor_tensor_network(
  network, (i, j, k) -> (j + 0.01 * randn(), k + 0.01 * randn())
)
@time lnz = contract_w_sweep(ltn; rank=256)
@info "lnZ of SweepContractor is", lnz
show(ITensors.timer)
