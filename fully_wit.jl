using CairoMakie, EntanglementWitnesses, Clarabel, JuMP, Combinatorics, StructuredLight

include("analytical_correlations.jl")

zs = LinRange(0, 2, 16)
wits = similar(zs)

theme = merge(
    theme_latexfonts(),
    Theme(
        fontsize=13.333333,
        linewidth=2,
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

    vs = reduce(vcat, [[_lg(rs, rs, l=l₂) * √dA], [_lg(rs, rs; p, l=l₁, w=1/√3) * √dA for p ∈ 0:2]])

    ax = Axis(fig[1, 1], xlabel=L"Z", ylabel="Optimal Witness")

    party_labels = Dict(
        "A|BCD" => L"\mathcal{A}|\mathcal{B}_0\mathcal{B}_1\mathcal{C}",
        "AB|CD" => L"\mathcal{A}\mathcal{B}_0|\mathcal{B}_1\mathcal{C}",
        "AC|BD" => L"\mathcal{A}\mathcal{B}_1|\mathcal{B}_0\mathcal{C}",
        "AD|BC" => L"\mathcal{A}\mathcal{C}|\mathcal{B}_0\mathcal{B}_1",
        "ABC|D" => L"\mathcal{A}\mathcal{B}_0\mathcal{B}_1|\mathcal{C}",
        "ABD|C" => L"\mathcal{A}\mathcal{B}_0\mathcal{C}|\mathcal{B}_1",
        "ACD|B" => L"\mathcal{A}\mathcal{B}_1\mathcal{C}|\mathcal{B}_0",
    )

    for m ∈ 2:2
        for part in partitions(1:length(vs), m)
            Threads.@threads for n ∈ eachindex(zs, wits)
                model = Model(Clarabel.Optimizer)
                set_attributes(model, "tol_feas" => 1e-4, "tol_gap_abs" => 1e-4)
                wits[n] = getWitness(FullyWit(), model, γ_matrix(zs[n], u, vs, NoDiffraction()), part)[1]
            end
            lines!(ax, zs, wits; label=party_labels[partition2label(part)])
        end
    end
    Legend(fig[:, end+1], ax)
    save("Plots/fully_wit.pdf", fig)
    fig
end
