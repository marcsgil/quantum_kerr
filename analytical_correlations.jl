using LinearAlgebra

function generalized_dot(v1, v2)
    conj(prod(v1)) * prod(v2)
end

struct Linear end

function f(Z, u, v1, v2, ::Linear)
    result = complex(zero(eltype(u)))
    for (u_i, v1_i, v2_i) in zip(u, v1, v2)
        result -= im * conj(v1_i * v2_i) * u_i^2 * Z
    end
    result
end

function h(Z, u, v1, v2, ::Linear)
    complex(zero(eltype(u)))
end

struct NoDiffraction end

function f(Z, u, v1, v2, ::NoDiffraction)
    sum(zip(u, v1, v2)) do (u, v1, v2)
        uz = u * cis(-Z * abs2(u))
        Z * generalized_dot((v1, v2), (uz, uz)) * (-im - Z * abs2(uz))
    end
end

function h(Z, u, v1, v2, ::NoDiffraction)
    sum(zip(u, v1, v2)) do (u, v1, v2)
        uz = u * cis(-Z * abs2(u))
        Z^2 * generalized_dot((v1, uz, uz), (v2, uz, uz))
    end
end

function γ(Z, u, v1, v2, approximation)
    2 * real(v1 ⋅ v2 / 2 + f(Z, u, v1, v2, approximation) + h(Z, u, v1, v2, approximation))
end

function γ_matrix(Z, u, vs, approximation)
    quadratures = reduce(vcat, [[v, im * v] for v ∈ vs])
    [γ(Z, u, v1, v2, approximation) for v1 ∈ quadratures, v2 ∈ quadratures]
end

function λ₋(Z, u, v, _g, approximation)
    1 + 2 * real(h(Z, u, v, v, approximation)) - 2 * abs(f(Z, u, v, v, approximation))
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

function _lg(args...; kwargs...)
    if :p ∈ keys(kwargs)
        p = Int(kwargs[:p])
    else
        p = 0
    end
    (-1)^p * lg(args...; kwargs...)
end