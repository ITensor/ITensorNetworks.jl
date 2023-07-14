include("apply_bp.jl")

opname = "Id"
# opname = "RandomUnitary"

@show opname

# graph = named_comb_tree
graph = named_grid

dims = (6, 6)

ψ_bp, mts_bp, ψ_vidal, mts_vidal = main(;
  seed=1234,
  opname,
  graph,
  dims,
  χ=2,
  nlayers=2,
  variational_optimization_only=false,
  regauge=false,
  reduced=true,
)

v = dims .÷ 2

sz_bp = @show expect_bp("Sz", v, ψ_bp, mts_bp)
sz_vidal = @show expect_bp("Sz", v, ψ_vidal, mts_vidal)
@show abs(sz_bp - sz_vidal) / abs(sz_vidal)

# Run BP again
mts_bp = belief_propagation(
  norm_network(ψ_bp),
  mts_bp;
  contract_kwargs=(; alg="exact"),
  niters=50,
  target_precision=1e-5,
)
mts_vidal = belief_propagation(
  norm_network(ψ_vidal),
  mts_vidal;
  contract_kwargs=(; alg="exact"),
  niters=50,
  target_precision=1e-5,
)

sz_bp = @show expect_bp("Sz", v, ψ_bp, mts_bp)
sz_vidal = @show expect_bp("Sz", v, ψ_vidal, mts_vidal)
@show abs(sz_bp - sz_vidal) / abs(sz_vidal)

ψ_symmetric, _ = symmetric_gauge(ψ_bp)

v⃗ⱼ = [v .+ (1, 0), v .- (1, 0), v .+ (0, 1), v .- (0, 1)]
ψ_bp_v = vertex_array(ψ_bp, v, v⃗ⱼ)
ψ_vidal_v = vertex_array(ψ_vidal, v, v⃗ⱼ)
ψ_symmetric_v = vertex_array(ψ_symmetric, v, v⃗ⱼ)

@show norm(abs.(ψ_bp_v) - abs.(ψ_vidal_v))
@show norm(abs.(ψ_bp_v) - abs.(ψ_symmetric_v))
@show norm(abs.(ψ_vidal_v) - abs.(ψ_symmetric_v))

inner_ψ_net = inner_network(ψ_bp, ψ_vidal)
norm_ψ_bp_net = norm_network(ψ_bp)
norm_ψ_vidal_net = norm_network(ψ_vidal)
seq = contraction_sequence(inner_ψ_net; alg="sa_bipartite")
@disable_warn_order begin
  inner_ψ = contract(inner_ψ_net; sequence=seq)[]
  norm_sqr_ψ_bp = contract(norm_ψ_bp_net; sequence=seq)[]
  norm_sqr_ψ_vidal = contract(norm_ψ_vidal_net; sequence=seq)[]
end
@show 1 - inner_ψ / (sqrt(norm_sqr_ψ_bp) * sqrt(norm_sqr_ψ_vidal))
@show log(inner_ψ) - (log(norm_sqr_ψ_bp) / 2 + log(norm_sqr_ψ_vidal) / 2)
