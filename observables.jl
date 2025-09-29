using LinearAlgebra

function projection(u, v, dA)
    (u ⋅ v) * dA
end

function expval_annihilation(α, β, v, dA)
    mapreduce(α -> (v ⋅ α) * dA, +, eachslice(α, dims=3)) / size(α, 3)
end

function expval_creation(α, β, v, dA)
    conjv = conj.(v)
    mapreduce(β -> (conjv ⋅ β) * dA, +, eachslice(β, dims=3)) / size(β, 3)
end

function expval_creation_annihilation(α, β, v1, v2, dA)
    conjv2 = conj.(v2)
    mapreduce((α, β) -> (v1 ⋅ α) * (conjv2 ⋅ β) * dA^2, +, eachslice(α, dims=3), eachslice(β, dims=3)) / size(α, 3)
end

function expval_creation_creation(α, β, v1, v2, dA)
    mapreduce(α -> (v1 ⋅ α) * (v2 ⋅ α) * dA^2, +, eachslice(α, dims=3)) / size(α, 3)
end

function correlation(α, β, v1, v2, dA)
    projection(v1, v2, dA) + real(
        expval_creation_creation(α, β, v1, v2, dA) - expval_annihilation(α, β, v1, dA) * expval_annihilation(α, β, v2, dA)
        +
        expval_creation_annihilation(α, β, v1, v2, dA) - expval_annihilation(α, β, v1, dA) * expval_creation(α, β, v2, dA)
    )
end