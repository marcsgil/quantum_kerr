using CairoMakie, EntanglementWitnesses, Clarabel, JuMP, Combinatorics

include("analytical_correlations.jl")

zs = LinRange(0, 10, 16)
wits = similar(zs)

theme = merge(
    theme_latexfonts(),
    Theme(
        linewidth=4,
        palette=(
            color=Makie.to_colormap(:Set2_7),
        ),
    )
)

with_theme(theme) do
    fig = Figure()

    l₀ = 0
    l₂ = 1
    l₁ = 2l₀ - l₂

    u = lg(rs, rs, l=l₀)

    ax = Axis(fig[1, 1], xlabel=L"Z", ylabel="Optimal Witness")

    for pmax ∈ 0:2
        vs = reduce(vcat, [[lg(rs, rs, l=l₂) * √dA], [lg(rs, rs; p, l=l₁, w=1/√3) * √dA for p ∈ 0:pmax]])
        Threads.@threads for n ∈ eachindex(zs, wits)
            model = Model(Clarabel.Optimizer)
            set_attributes(model, "tol_feas" => 1e-4, "tol_gap_abs" => 1e-4)
            wits[n] = getWitness(MultiWit(), model, γ_matrix(zs[n], u, vs, NoDiffraction()))[1]
        end
        lines!(ax, zs, wits; label=L"p_{\text{max}} = %$pmax")

    end
    Legend(fig[:, end+1], ax)
    fig
end