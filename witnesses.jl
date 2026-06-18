using StructuredLight, CairoMakie, LinearAlgebra, EntanglementWitnesses, Clarabel, JuMP
include("observables.jl")

L = 8
lengths = (L, L)
N = 128
dr = L / N
rs = LinRange(-L / 2, L / 2 - dr, N)
α₀ = 5e4

α = α₀ * lg(rs, rs, l=0)

g = 1e-7
dA = dr^2

function f(z, v1, v2, α, g)
    Z = g * z / 4
    αz = α .* cis.(-Z * abs2.(α))

    (-im * Z * (v1 .* v2) ⋅ (αz .^ 2) - Z .^ 2 * (v1 .* v2 .* αz) ⋅ (αz .^ 3))
end

function h(z, v1, v2, α, g)
    Z = g * z / 4
    αz = α .* cis.(-Z * abs2.(α))
    Z .^ 2 * (v1 .* αz .^ 2) ⋅ (v2 .* αz .^ 2)
end

function γ(z, v1, v2, α, g)
    2 * real(v1 ⋅ v2 / 2 + f(z, v1, v2, α, g) + h(z, v1, v2, α, g))
end

function γ(z, vs, α, g)
    quadratures = reduce(vcat, [[v, -im * v] for v ∈ vs])
    [γ(z, v1, v2, α, g) for v1 ∈ quadratures, v2 ∈ quadratures]
end

v0 = lg(rs, rs, l=0)
α = α₀ * v0
##
# Opposite l

lmax = 4
pmax = 2


@show g * z_max / 4dA
@show (g * z_max / 4)^2 * maximum(abs2, α) / dA

wits = similar(zs)


with_theme(theme_latexfonts()) do
    fig = Figure(; size=(800, 400))
    for l ∈ 1:lmax
        ax = Axis(fig[(l-1)÷2, (l-1)%2], xlabel=L"z/z_r", ylabel="Optimal Witness", title=L"l=%$l")
        zs = LinRange(0, 0.05 * sqrt(l), 16)
        for q ∈ 0:pmax
            @show l, q
            vs = reduce(vcat, [[lg(rs, rs; l=l, p), lg(rs, rs; l=-l, p)] for p ∈ 0:q]) .* √dA
            for (n, z) ∈ enumerate(zs)
                model = Model(Clarabel.Optimizer)
                set_attributes(model, "tol_feas" => 1e-4, "tol_gap_abs" => 1e-4)
                wits[n] = getWitness(MultiWit(), model, γ(z, vs, α, g))[1]
            end
            lines!(ax, zs, wits, linewidth=4, label=L"p_{\text{max}} = %$q")

            if l == lmax && q == pmax
                Legend(fig[:, -1], ax)
            end
        end
    end
    fig
end
##
zs = LinRange(0, 0.05, 16)

with_theme(theme_latexfonts()) do
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel=L"z/z_r", ylabel="Optimal Witness")
    zs = LinRange(0, 0.05, 16)
    vs = [lg(rs, rs; l=0), lg(rs, rs; l=1), lg(rs, rs; l=-1)] .* √dA
    wits_BC = similar(zs)
    wits_AB = similar(zs)
    for (n, z) ∈ enumerate(zs)
        model = Model(Clarabel.Optimizer)
        set_attributes(model, "tol_feas" => 1e-4, "tol_gap_abs" => 1e-4)

        model = Model(Clarabel.Optimizer)
        wits_BC[n] = getWitness(FullyWit(), model, γ(z, vs, α, g), [[1], [2, 3]])[1]
        model = Model(Clarabel.Optimizer)
        wits_AB[n] = getWitness(FullyWit(), model, γ(z, vs, α, g), [[1, 2], [3]])[1]
    end
    lines!(ax, zs, wits_BC, linewidth=4, label="A|BC")
    lines!(ax, zs, wits_AB, linewidth=4, label="AB|C")
    axislegend(ax, position = :lt)
    #save("Plots/|l|=1,l=0_partition.png", fig)
    fig
end
##
lmax = 4
pmax = 2


@show g * z_max / 4dA
@show (g * z_max / 4)^2 * maximum(abs2, α) / dA


with_theme(theme_latexfonts()) do
    fig = Figure(; size=(800, 400))
    for l ∈ 1:lmax
        ax = Axis(fig[(l-1)÷2, (l-1)%2], xlabel=L"z/z_r", ylabel="Optimal Witness", title=L"l=%$l")
        zs = LinRange(0, 0.05 * sqrt(l), 16)
        wits = similar(zs)
        for q ∈ 0:pmax
            @show l, q
            # vs = reduce(vcat, [[lg(rs, rs; l=-l)], [lg(rs, rs; l=l, p) for p ∈ 0:q]]) .* √dA
            vs = [lg(rs, rs; l=l, p) for p ∈ 0:q] .* √dA
            for (n, z) ∈ enumerate(zs)
                model = Model(Clarabel.Optimizer)
                set_attributes(model, "tol_feas" => 1e-4, "tol_gap_abs" => 1e-4)
                wits[n] = getWitness(MultiWit(), model, γ(z, vs, α, g))[1]
            end
            lines!(ax, zs, wits, linewidth=4, label=L"p_{\text{max}} = %$q")

            if l == lmax && q == pmax
                Legend(fig[:, -1], ax)
            end
        end
    end
    fig
end