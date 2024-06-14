module ITensorNetworksAdaptExt
using Adapt: Adapt, adapt
using ITensorNetworks: AbstractITensorNetwork, map_vertex_data_preserve_graph
function Adapt.adapt_structure(to, tn::AbstractITensorNetwork)
  # TODO: Define and use:
  #
  # @preserve_graph map_vertex_data(adapt(to), tn)
  #
  # or just:
  #
  # @preserve_graph map(adapt(to), tn)
  return map_vertex_data_preserve_graph(adapt(to), tn)
end
end
