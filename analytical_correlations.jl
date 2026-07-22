using LinearAlgebra

function generalized_dot(v1, v2)
    conj(prod(v1)) * prod(v2)
end

struct Linear end

function f(Z, u, v1, v2, ::Linear)
    result = complex(zero(eltype(α)))
    for (a, b, c) in zip(u, v1, v2)
        result -= im * conj(b * c) * a^2 * g * Z
    end
    result
end

function h(z, α, v1, v2, ::Linear)
    complex(zero(eltype(α)))
end

struct NoDiffraction end

function f(Z, u, v1, v2, ::NoDiffraction)
    sum(zip(u, v1, v2)) do (u, v1, v2)
        uz = u * cis(-Z * abs2(u))
        Z * generalized_dot((v1, v2), (u, u)) * (-im + - Z * abs2(uz))
    end
end

function h(Z, u, v1, v2, ::NoDiffraction)
    sum(zip(u, v1, v2)) do (u, v1, v2)
        Z^2 * generalized_dot((v1, u, u), (v2, u, u))
    end
end

function γ(Z, u, v1, v2, approximation)
    2 * real(v1 ⋅ v2 / 2 + f(Z, u, v1, v2, approximation) + h(Z, u, v1, v2, approximation))
end

function γ_matrix(Z, u, vs, approximation)
    quadratures = reduce(vcat, [[v, -im * v] for v ∈ vs])
    [γ(Z, u, v1, v2, approximation) for v1 ∈ quadratures, v2 ∈ quadratures]
end

function λ₋(z, α, v, g, approximation)
    1 / 2 + real(h(z, α, v, v, approximation)) + abs(f(z, α, v, v, approximation))
end

function duan(Z, u, v1, v2, approximation)
    (1 + real(h(Z, u, v1, v1, approximation) + h(Z, u, v2, v2, approximation))
     -
     2 * abs(f(Z, u, v1, v2, approximation)))
end

function partition2label(part, alphabet=collect('A':'Z'))
    mapreduce(part -> prod(x->alphabet[x], part), (str1, str2) -> str1 * "|" * str2, part)
end

L = 8
N = 128
dr = L / N
rs = LinRange(-L / 2, L / 2 - dr, N)

dA = dr^2