default_index_map = prime
default_inv_index_map = noprime

struct QuadraticFormNetwork{V,FormNetwork<:BilinearFormNetwork{V},IndexMap,InvIndexMap} <:
       AbstractFormNetwork{V}
  formnetwork::FormNetwork
  dual_index_map::IndexMap
  dual_inv_index_map::InvIndexMap
end

bilinear_formnetwork(qf::QuadraticFormNetwork) = qf.formnetwork
function QuadraticFormNetwork(
  operator::AbstractITensorNetwork,
  bra::AbstractITensorNetwork,
  ket::AbstractITensorNetwork;
  dual_index_map=default_index_map,
  dual_inv_index_map=default_inv_index_map,
  kwargs...,
)
  return QuadraticFormNetwork(
    BilinearFormNetwork(operator, bra, ket; kwargs...), dual_index_map, dual_inv_index_map
  )
end

#Needed for implementation, forward from bilinear form
for f in [
  :operator_vertex_suffix,
  :bra_vertex_suffix,
  :ket_vertex_suffix,
  :tensornetwork,
  :data_graph,
  :data_graph_type,
]
  @eval begin
    function $f(qf::QuadraticFormNetwork, args...; kwargs...)
      return $f(bilinear_formnetwork(qf), args...; kwargs...)
    end
  end
end

dual_index_map(qf::QuadraticFormNetwork) = qf.dual_index_map
dual_inv_index_map(qf::QuadraticFormNetwork) = qf.dual_inv_index_map
function copy(qf::QuadraticFormNetwork)
  return QuadraticFormNetwork(
    copy(bilinear_formnetwork(qf)), dual_index_map(qf), dual_inv_index_map(qf)
  )
end

function QuadraticFormNetwork(
  operator::AbstractITensorNetwork,
  ket::AbstractITensorNetwork;
  dual_index_map=default_index_map,
  dual_inv_index_map=default_inv_index_map,
  kwargs...,
)
  bra = map_inds(dual_index_map, dag(ket))
  blf = BilinearFormNetwork(operator, bra, ket; kwargs...)
  return QuadraticFormNetwork(blf, dual_index_map, dual_inv_index_map)
end

function QuadraticFormNetwork(
  ket::AbstractITensorNetwork;
  dual_index_map=default_index_map,
  dual_inv_index_map=default_inv_index_map,
  kwargs...,
)
  s = siteinds(ket)
  operator_inds = union_all_inds(s, dual_index_map(s; links=[]))
  operator = delta_network(operator_inds)
  bra = map_inds(dual_index_map, dag(ket))
  blf = BilinearFormNetwork(operator, bra, ket; kwargs...)
  return QuadraticFormNetwork(blf, dual_index_map, dual_inv_index_map)
end

function bra_ket_vertices(qf::QuadraticFormNetwork, state_vertices::Vector)
  return vcat(bra_vertices(qf, state_vertices), ket_vertices(qf, state_vertices))
end

function update(qf::QuadraticFormNetwork, state_vertex, state::ITensor)
  qf = copy(qf)
  state_inds = inds(state)
  state_dag = replaceinds(dag(state), state_inds, dual_index_map(qf).(state_inds))
  # TODO: Maybe add a check that it really does preserve the graph.
  setindex_preserve_graph!(tensornetwork(qf), state, ket_vertex_map(qf)(state_vertex))
  setindex_preserve_graph!(tensornetwork(qf), state_dag, bra_vertex_map(qf)(state_vertex))
  return qf
end
