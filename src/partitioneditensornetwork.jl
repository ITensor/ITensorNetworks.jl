const PartitionedITensorNetwork{V, PV} = PartitionedGraph

NamedGraphs.parent_graph(pitn::PartitionedITensorNetwork) = NamedGraphs.parent_graph(underlying_graph(unpartitioned_graph(pitn)))
