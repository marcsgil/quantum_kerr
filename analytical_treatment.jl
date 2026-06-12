using StructuredLight, CairoMakie, LinearAlgebra

L = 8
lengths = (L, L)
N = 128
dr = L / N
rs = LinRange(-L / 2, L / 2 - dr, N)
α₀ = 9e3

α = α₀ * lg(rs, rs, l=0)

g = 4e-7
dA = dr^2

z_max = 0.4

zs = LinRange(0, z_max, 256)

@show g * z_max / 4dA
@show (g * z_max / 4)^2 * maximum(abs2, α) / dA

function f(z, v1, v2, α, g, dA)
    Z = g * z / 4
    αz = α .* cis.(-Z * abs2.(α))
    (-im * Z * (v1 .* v2) ⋅ (αz .^ 2) - Z .^ 2 * (v1 .* v2 .* αz) ⋅ (αz .^ 3)) * dA
end

function h(z, v1, v2, α, g, dA)
    Z = g * z / 4
    αz = α .* cis.(-Z * abs2.(α))
    Z .^ 2 * (v1 .* αz .^ 2) ⋅ (v2 .* αz .^ 2) * dA
end

function λ₋(z, v, α, g, dA)
    1 // 2 + real(h(z, v, v, α, g, dA)) - abs(f(z, v, v, α, g, dA))
end

function duan(z, v1, v2, α, g, dA)
    2 * (1 + real(h(z, v1, v1, α, g, dA) + h(z, v2, v2, α, g, dA))) - 4abs(f(z, v1, v2, α, g, dA))
end

v = lg(rs, rs, l=0)
v1 = lg(rs, rs, l=1)
v2 = lg(rs, rs, l=-1)

λ₋s = map(z -> λ₋(z, v, α, g, dA), zs)
duans = map(z -> duan(z, v1, v2, α, g, dA), zs)


lines(zs, λ₋s)
lines(zs, duans)