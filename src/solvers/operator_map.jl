function optimal_map(P::ProjTTN, ψ)
    envs = [environment(P, e) for e in incident_edges(P)]
    site_ops = [operator(P)[s] for s in sites(P)]
    contract_list = [envs..., site_ops..., ψ]
    sequence = contraction_sequence(contract_list; alg = "optimal")
    Pψ = contract(contract_list; sequence)
    return noprime(Pψ)
end
