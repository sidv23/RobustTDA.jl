## Log Persistence Diagram
import Base: log
import Ripserer: PersistenceDiagram, PersistenceInterval

Base.:log(X::Nothing) = nothing

Base.:log(S::Ripserer.Simplex) = S.birth == 0 ? Simplex{Ripserer.dim(S)}(S.index, -6.) : Simplex{Ripserer.dim(S)}(S.index, log(S.birth))

Base.:log(X::NamedTuple) = X.birth_simplex == 0 ? (; birth_simplex = -6., death_simplex = log(X.death_simplex)) : (; birth_simplex = log(X.birth_simplex), death_simplex = log(X.death_simplex))

Base.:log(I::Ripserer.PersistenceInterval) = I[1] == 0 ? PersistenceInterval([-6., log(I[2])]..., log(I.meta)) : PersistenceInterval(log.(I)..., log(I.meta))

Base.:log(D::Ripserer.PersistenceDiagram) = PersistenceDiagram(log.(D.intervals), D.meta)

Base.:log(PD::Vector{Ripserer.PersistenceDiagram}) = map(x -> log(x), PD)
