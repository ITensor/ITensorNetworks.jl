import Graphs: dst, src
import NamedGraphs.GraphsExtensions:
  default_root_vertex, post_order_dfs_edges, post_order_dfs_vertices

function post_order_dfs_plan(
  graph; nsites, root_vertex=default_root_vertex(graph), sweep_kwargs...
)
  if nsites == 1
    vertices = post_order_dfs_vertices(graph, root_vertex)
    fwd_sweep = [([v], sweep_kwargs) for v in vertices]
  elseif nsites == 2
    edges = post_order_dfs_edges(graph, root_vertex)
    fwd_sweep = [([src(e), dst(e)], sweep_kwargs) for e in edges]
  end
  return fwd_sweep
end

function post_order_dfs_sweep(args...; kws...)
  fwd_sweep = post_order_dfs_plan(args...; kws...)
  rev_sweep = [(reverse(reg_kws[1]), reg_kws[2]) for reg_kws in reverse(fwd_sweep)]
  return [fwd_sweep..., rev_sweep...]
end
