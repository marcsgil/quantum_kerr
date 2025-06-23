using GeneralizedGrossPitaevskii, StructuredLight, CairoMakie, CUDA

function dispersion(k, param)
    factor = param.ħ / (2 * param.m)
    @SVector [factor * (k[1]^2 + k[2]^2), -factor * (k[1]^2 + k[2]^2)]
end

function nonlinearity(u, param)
    factor = param.g * u[1] * u[2] / param.ħ / param.ΔV
    @SVector [factor, -factor]
end

function position_noise_func(u, r, param)
    factor = √(im * param.g / param.ħ)
    @SVector [factor * u[1], -conj(factor) * u[2]]
end

L = 10f0
lengths = (L, L)
N = 256
Δr = L / N
ΔV = Δr^2
rs = StepRangeLen(0, Δr, N)

_u0 = lg(rs .- N * Δr / 2, rs .- N * Δr / 2, l=1) |> cu
u0 = stack(_u0 for _ ∈ 1:10^2)

ħ = 1f0
m = 1f0
g = -1f-2

U0 = (u0, conj(u0))
noise_prototype = similar.(U0, Float32)
param = (; L, N, Δr, ΔV, ħ, m, g)

prob = GrossPitaevskiiProblem(U0, lengths; dispersion, nonlinearity, position_noise_func, noise_prototype, param)
alg = StrangSplitting()
tspan = (0, 0.5f0)
nsaves = 1
dt = 1f-2


ts, sol = solve(prob, alg, tspan; dt, nsaves, save_start=false)

visualize(abs2.(sol[1][:, :, 1]))

I = dropdims(mean(map((x, y) -> real(x .* conj(y)), sol[1], sol[2]), dims=3), dims=(3, 4))

visualize(I)