using Graphs: add_edge!, add_vertex!
using NamedGraphs: NamedGraph

"""
  build_tree

  Make a tree with central vertex (0,0) and
  nbranch branches of nbranch_sites each.
"""
function build_tree(; nbranch=3, nbranch_sites=3)
  g = NamedGraph()
  add_vertex!(g, (0, 0))
  for branch in 1:nbranch, site in 1:nbranch_sites
    add_vertex!(g, (branch, site))
  end
  for branch in 1:nbranch
    add_edge!(g, (0, 0)=>(branch, 1))
    for site in 2:nbranch_sites
      add_edge!(g, (branch, site-1)=>(branch, site))
    end
  end
  return g
end
