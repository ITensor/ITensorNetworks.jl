using Graphs: dst, src
using NamedGraphs.GraphsExtensions: default_root_vertex

function euler_sweep(graph; nsites, root_vertex=default_root_vertex(graph), sweep_kwargs...)
  sweep_kwargs = (; nsites, root_vertex, sweep_kwargs...)

  if nsites == 1
    vertices = euler_tour_vertices(graph, root_vertex)
    sweep = [[v] => sweep_kwargs for v in vertices]
  elseif nsites == 2
    edges = euler_tour_edges(graph, root_vertex)
    sweep = [[src(e), dst(e)] => sweep_kwargs for e in edges]
  end
  return sweep
end
