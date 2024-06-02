
using ITensors: Trotter
using ITensorNetworks: norm_sqr_network, update
using SplitApplyCombine: group

function imaginary_time_evo(s::IndsNetwork, ψ::ITensorNetwork, model::Function, dbetas::Vector{<:Tuple}, nbetas::Int64; model_params,
    bp_update_kwargs = (; maxiter = 10, tol = 1e-10), apply_kwargs = (; cutoff = 1e-12, maxdim = 10))
    ψ = copy(ψ)
    g = underlying_graph(ψ)
    
    ℋ =model(g; model_params...)
    ψψ = norm_sqr_network(ψ)
    bpc = BeliefPropagationCache(ψψ, group(v->v[1], vertices(ψψ)))
    bpc = update(bpc; bp_update_kwargs...)
    println("Starting Imaginary Time Evolution")
    β = 0
    for (i, period) in enumerate(dbetas)
        nbetas, dβ = first(period), last(period)
        println("Entering evolution period $i , β = $β, dβ = $dβ")
        U = exp(- dβ * ℋ, alg = Trotter{1}())
        gates = Vector{ITensor}(U, s)
        for i in 1:nbetas
            for gate in gates
                ψ, bpc = BP_apply(gate, ψ, bpc; apply_kwargs...)
            end
            β += dβ
            bpc = update(bpc; bp_update_kwargs...)
        end
        e = sum(expect(ψ, ℋ; alg = "bp"))
        println("Energy is $e")
    end

    return ψ
end