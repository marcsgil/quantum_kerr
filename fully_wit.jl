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

    l₀ = 2
    l₂ = 1
    l₁ = 2l₀ - l₂

    u = lg(rs, rs, l=l₀)

    vs = reduce(vcat, [[lg(rs, rs, l=l₂) * √dA], [lg(rs, rs; p, l=l₁, w=1/√3) * √dA for p ∈ 0:2]])

    ax = Axis(fig[1, 1], xlabel=L"Z", ylabel="Optimal Witness")

    for m ∈ 2:2
        for part in partitions(1:length(vs), m)
            Threads.@threads for n ∈ eachindex(zs, wits)
                model = Model(Clarabel.Optimizer)
                set_attributes(model, "tol_feas" => 1e-4, "tol_gap_abs" => 1e-4)
                wits[n] = getWitness(FullyWit(), model, γ_matrix(zs[n], u, vs, NoDiffraction()), part)[1]
            end
            lines!(ax, zs, wits; label=partition2label(part))
        end
    end
    Legend(fig[:, end+1], ax)
    save("Plots/fully_wit.pdf", fig)
    fig
end