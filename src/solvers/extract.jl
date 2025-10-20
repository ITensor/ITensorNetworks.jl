function extract!(region_iter::RegionIterator; subspace_algorithm = "nothing")
    prob = problem(region_iter)
    region = current_region(region_iter)

    psi = orthogonalize(state(prob), region)
    local_state = prod(psi[v] for v in region)

    prob.state = psi

    _, local_state = subspace_expand!(region_iter, local_state; subspace_algorithm)
    shifted_operator = position(operator(prob), state(prob), region)

    prob.operator = shifted_operator

    return region_iter, local_state
end
