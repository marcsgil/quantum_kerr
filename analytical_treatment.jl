using StructuredLight, CairoMakie, LinearAlgebra

L = 8
lengths = (L, L)
N = 128
dr = L / N
rs = LinRange(-L / 2, L / 2 - dr, N)
α₀ = 3e4

α = α₀ * lg(rs, rs, l=0)

g = 4e-7
dA = dr^2
G = g / (4 * dA)

z_max = 0.4

zs = LinRange(0, z_max, 256)

@show G * z_max

function f(z, v1, v2, α, g, dA)
    αz = α .* cis.(-g * z * abs2.(α) / 4)
    -im * g * z / 4 * (v1 .* v2) ⋅ (αz.^2) * dA
end

v = lg(rs, rs, l=0)
v1 = lg(rs, rs, l=0)
v2 = lg(rs, rs, l=-0)

fs = map(z-> f(z, v1, v2, α, g, dA), zs)

lines(zs, abs.(fs))
# lines(zs, angle.(fs))