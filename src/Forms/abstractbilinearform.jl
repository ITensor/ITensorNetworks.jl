abstract type AbstractForm{V,BraMap,KetMap,OperatorMap,SpaceMap} <:
              AbstractITensorNetwork{V} end

#Needed for interface
bra_map(f::AbstractForm) = not_implemented()
ket_map(f::AbstractForm) = not_implemented()
operator_map(f::AbstractForm) = not_implemented()
space_map(f::AbstractForm) = not_implemented()
tensornetwork(f::AbstractForm) = not_implemented()
copy(f::AbstractForm) = not_implemented()

bra(f::AbstractForm) = induced_subgraph(f, collect(values(bra_map(f))))
ket(f::AbstractForm) = induced_subgraph(f, collect(values(ket_map(f))))
operator(f::AbstractForm) = induced_subgraph(f, collect(values(operator_map(f))))
