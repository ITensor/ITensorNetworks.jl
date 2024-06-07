using ITensors: scalartype
using ITensorNetworks: ket_vertices, bra_vertices
using ITensorNetworks.ITensorsExtensions: map_eigvals


function bp_extracter(ψ::AbstractITensorNetwork, ψAψ_bpcs::Vector{<:BeliefPropagationCache}, ψIψ_bpc::BeliefPropagationCache, region;
    regularization = 10*eps(scalartype(ψ)))
    form_network = unpartitioned_graph(partitioned_tensornetwork(ψIψ_bpc))
    form_ket_vertices, form_bra_vertices = ket_vertices(form_network, region), bra_vertices(form_network, region)
    ∂ψAψ_bpc_∂rs = [environment(ψAψ_bpc, [form_ket_vertices; form_bra_vertices]) for ψAψ_bpc in ψAψ_bpcs]
    state = prod(ψ[v] for v in region)
    messages = environment(ψIψ_bpc, partitionvertices(ψIψ_bpc, form_ket_vertices))
    f_sqrt = sqrt ∘ (x -> x + regularization)
    f_inv_sqrt = inv ∘ sqrt ∘ (x -> x + regularization)
    sqrt_mts = [map_eigvals(f_sqrt, mt, inds(mt)[1], inds(mt)[2]; ishermitian=true) for mt in messages]
    inv_sqrt_mts = [map_eigvals(f_inv_sqrt, mt, inds(mt)[1], inds(mt)[2]; ishermitian=true) for mt in messages]
  
    return state, ∂ψAψ_bpc_∂rs, sqrt_mts, inv_sqrt_mts
  end