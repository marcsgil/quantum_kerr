using CairoMakie, EntanglementWitnesses, Clarabel, JuMP, Combinatorics, StructuredLight

include("analytical_correlations.jl")

zs = LinRange(0, 2, 16)
wits = similar(zs)

theme = merge(
    theme_latexfonts(),
    Theme(
        fontsize=13.333333,
        linewidth=3,
        palette=(
            color=Makie.to_colormap(:Set2_7),
        ),
    )
)

with_theme(theme) do
    fig = Figure(size=(327, 245))

    l₀ = 2
    l₂ = 1
    l₁ = 2l₀ - l₂

    u = _lg(rs, rs, l=l₀)

    ax = Axis(fig[1, 1], xlabel=L"Z", ylabel="Optimal Witness")

    for pmax ∈ 0:3
        vs = reduce(vcat, [[_lg(rs, rs, l=l₂) * √dA], [_lg(rs, rs; p, l=l₁, w=1/√3) * √dA for p ∈ 0:pmax]])
        Threads.@threads for n ∈ eachindex(zs, wits)
            model = Model(Clarabel.Optimizer)
            set_attributes(model, "tol_feas" => 1e-4, "tol_gap_abs" => 1e-4)
            wits[n] = getWitness(MultiWit(), model, γ_matrix(zs[n], u, vs, NoDiffraction()))[1]
        end
        lines!(ax, zs, wits; label=L"P = %$pmax")

    end
    Legend(fig[:, end+1], ax)
    save("Plots/multi_wit.pdf", fig)
    fig
end
