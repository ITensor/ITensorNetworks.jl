using NamedGraphs: edgetype

function insert!(region_iter, local_tensor; normalize = false, set_orthogonal_region = true)
    prob = problem(region_iter)

    region = current_region(region_iter)
    psi = copy(state(prob))
    if length(region) == 1
        C = local_tensor
    elseif length(region) == 2
        e = edgetype(psi)(first(region), last(region))
        indsTe = inds(psi[first(region)])
        tags = ITensors.tags(psi, e)

        trunc_kwargs = truncation_parameters(
            region_iter.which_sweep;
            region_kwargs(factorize, region_iter)...
        )
        U, C, spectrum = factorize(
            local_tensor, indsTe; tags, trunc_kwargs...
        )

        psi[first(region)] = U
        prob = set_truncation_info!(prob; spectrum)
    else
        error("Region of length $(length(region)) not currently supported")
    end
    v = last(region)
    psi[v] = C
    psi = set_orthogonal_region ? set_ortho_region(psi, [v]) : psi
    if normalize
        psi[v] = psi[v] / norm(psi[v])
    end

    prob.state = psi

    return region_iter
end
