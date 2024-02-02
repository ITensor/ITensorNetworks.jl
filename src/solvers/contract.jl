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
  P = projected_operator![]
  return contract_ket(P, ITensor(true)), (;)
end
