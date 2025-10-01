using GeneralizedGrossPitaevskii, StructuredLight, CairoMakie, CUDA, Statistics
includet("observables.jl")

function dispersion(k, param)
    factor = (k[1]^2 + k[2]^2) / (2 * param.k)
    @SVector [factor, -factor]
end

function nonlinearity(u, param)
    factor = param.G * u[1] * u[2]
    @SVector [factor, -factor]
end

function position_noise_func(u, r, param)
    factor = √(im * param.G)
    @SVector [factor * u[1], -conj(factor) * u[2]]
end

L = 5f-3
lengths = (L, L)
N = 256
dr = L / N
dA = dr^2
rs = LinRange(-L / 2, L / 2 - dr, N)

_u0 = lg(rs, rs, l=1, w=1f-3)
u0 = stack(_u0 for _ ∈ 1:128) |> cu

λ = 633f-9
k = 2π / λ
g = 2f-5

G = g / k / dr^2

U0 = (u0, conj(u0))
noise_prototype = similar.(U0, Float32)
param = (; k, G)

prob = GrossPitaevskiiProblem(U0, lengths; dispersion, nonlinearity, position_noise_func, noise_prototype, param)
alg = StrangSplitting()
tspan = (0, 2f-2)
nsaves = 1
dt = 2f-5


ts, sol = solve(prob, alg, tspan; dt, nsaves, save_start=false)


U0[1]

sol[1]
##
save_animation(abs2.(sol[1]), "test.mp4")
##
visualize(abs2.(sol[1][:, :, end]))

intensity = dropdims(mean(map((x, y) -> real(x .* y), sol[1], sol[2]), dims=3), dims=(3, 4))

visualize(intensity)
##
_u0 = lg(rs, rs, l=1, w=1f-3)
u0 = stack(_u0 for _ ∈ 1:128) |> cu
U0 = (u0, conj(u0))
prob = GrossPitaevskiiProblem(U0, lengths; dispersion, nonlinearity, position_noise_func, noise_prototype, param)

v = lg(rs, rs, l=1, w=1f-3) |> cu
v1 = lg(rs, rs, l=0, w=1f-3) |> cu
v2 = lg(rs, rs, l=2, w=1f-3) |> cu

observables = (
    (α, β) -> expval_annihilation(α, β, v, dA),
    (α, β) -> correlation(α, β, v1, v2, dA)
)

ts, observables_vals = step_evolution(prob, 2f-2, observables; dt, nsaves=128)

with_theme(theme_latexfonts()) do 
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, ts, real.(observables_vals[1, :]), label="Re")
    lines!(ax, ts, imag.(observables_vals[1, :]), label="Im")
    fig
end

with_theme(theme_latexfonts()) do 
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, ts, real.(observables_vals[2, :]), label="Re")
    fig
end