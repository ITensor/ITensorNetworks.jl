# TODO: we may want to move these to `DataGraphs.jl`
for f in [:_root, :_is_rooted, :_is_rooted_directed_binary_tree]
  @eval begin
    function $f(graph::AbstractDataGraph, args...; kwargs...)
      return $f(underlying_graph(graph), args...; kwargs...)
    end
  end
end
