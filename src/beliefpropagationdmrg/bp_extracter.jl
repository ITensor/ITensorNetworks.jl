using ITensors: scalartype
using ITensorNetworks: ket_vertices, bra_vertices, tensornetwork
using ITensorNetworks.ITensorsExtensions: map_eigvals


function bp_extracter(ψ::AbstractITensorNetwork, ψOψ_bpcs::Vector{<:BeliefPropagationCache}, ψIψ_bpc::BeliefPropagationCache, region;
    regularization = 10*eps(scalartype(ψ)), ishermitian = true)

    form_network = tensornetwork(ψIψ_bpc)
    form_ket_vertices, form_bra_vertices = ket_vertices(form_network, region), bra_vertices(form_network, region)

    ∂ψOψ_bpc_∂rs = environment.(ψOψ_bpcs, ([form_ket_vertices; form_bra_vertices], ))
    state = prod(ψ[v] for v in region)
    messages = environment(ψIψ_bpc, partitionvertices(ψIψ_bpc, form_ket_vertices))
    f_sqrt = sqrt ∘ (x -> x + regularization)
    f_inv_sqrt = inv ∘ sqrt ∘ (x -> x + regularization)
    sqrt_mts = map_eigvals.((f_sqrt, ), messages, first.(inds.(messages)), last.(inds.(messages)); ishermitian)
    inv_sqrt_mts = map_eigvals.((f_inv_sqrt, ), messages, first.(inds.(messages)), last.(inds.(messages)); ishermitian)
    
    return state, ∂ψOψ_bpc_∂rs, sqrt_mts, inv_sqrt_mts
  end