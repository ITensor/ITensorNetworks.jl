using ..BaseExtensions: maybe_real, to_tuple
using Graphs: dst, edges, src
using ITensors.LazyApply: Applied, Prod, Scaled, Sum
using ITensors.Ops: Op, Ops
using ITensors: ITensors
using SplitApplyCombine: group

# TODO: Rename this `replace_sites`?
# TODO: Use `fmap`, `deepmap`, `treemap`?
function replace_vertices(f, ∑o::Sum)
    return Sum(map(oᵢ -> replace_vertices(f, oᵢ), Ops.terms(∑o)))
end

function replace_vertices(f, ∏o::Prod)
    return Prod(map(oᵢ -> replace_vertices(f, oᵢ), Ops.terms(∏o)))
end

function replace_vertices(f, o::Scaled)
    return maybe_real(Ops.coefficient(o)) * replace_vertices(f, Ops.argument(o))
end

set_sites(o::Op, sites) = Op(Ops.which_op(o), sites...; Ops.params(o)...)

function replace_vertices(f, o::Op)
    return set_sites(o, f.(Ops.sites(o)))
end

## function replace_vertices(o::Union{Op,Applied}, vertex_map)
##   return replace_vertices(v -> get(vertex_map, v, v), o)
## end

function group_terms(ℋ::Sum, g)
    grouped_terms = group(ITensors.terms(ℋ)) do t
        findfirst(edges(g)) do e
            return to_tuple.(ITensors.sites(t)) ⊆ [src(e), dst(e)]
        end
    end
    return Sum(collect(sum.(grouped_terms)))
end
