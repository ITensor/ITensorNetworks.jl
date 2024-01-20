function contract_updater(
    init;
    state!,
    projected_operator!,
    outputlevel,
    which_sweep,
    sweep_plan,
    which_region_update,
    region_kwargs,
    updater_kwargs,
    )
    v = ITensor(1.0)
    projected_operator = projected_operator![]
    for j in sites(projected_operator)
      v *= projected_operator.psi0[j]
    end
    Hpsi0 = contract(projected_operator, v)
    return Hpsi0, (;)
  end