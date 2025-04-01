using ..BaseExtensions: maybe_real, to_tuple
using Graphs: dst, edges, src
using QuantumOperatorAlgebra: Applied, Op, Prod, Scaled, Sum, terms, argument, params, which_op, sites
using SplitApplyCombine: group

# TODO: Rename this `replace_sites`?
# TODO: Use `fmap`, `deepmap`, `treemap`?
function replace_vertices(f, ∑o::Sum)
  return Sum(map(oᵢ -> replace_vertices(f, oᵢ), terms(∑o)))
end

function replace_vertices(f, ∏o::Prod)
  return Prod(map(oᵢ -> replace_vertices(f, oᵢ), terms(∏o)))
end

function replace_vertices(f, o::Scaled)
  return maybe_real(coefficient(o)) * replace_vertices(f, argument(o))
end

set_sites(o::Op, sites) = Op(which_op(o), sites...; params(o)...)

function replace_vertices(f, o::Op)
  return set_sites(o, f.(sites(o)))
end

## function replace_vertices(o::Union{Op,Applied}, vertex_map)
##   return replace_vertices(v -> get(vertex_map, v, v), o)
## end

function group_terms(ℋ::Sum, g)
  grouped_terms = group(terms(ℋ)) do t
    findfirst(edges(g)) do e
      to_tuple.(sites(t)) ⊆ [src(e), dst(e)]
    end
  end
  return Sum(collect(sum.(grouped_terms)))
end
