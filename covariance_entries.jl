using StructuredLight, Integrals, CairoMakie

function f(r, param)
    p1, p2, p3, p4, l1, l2, l3, l4, w1, w2, w3, w4 = param
    l1 + l2 != l3 + l4 && return zero(r)
    real(lg(r, 0, p=p1, l=l1, w=w1) * lg(r, 0, p=p2, l=l2, w=w2) *
         lg(r, 0, p=p3, l=l3, w=w3) * lg(r, 0, p=p4, l=l4, w=w4))
end

function four_mode_overlap(p1, p2, p3, p4, l1, l2, l3, l4, w1, w2, w3, w4)
    param = (p1, p2, p3, p4, l1, l2, l3, l4, w1, w2, w3, w4)
    domain = (0, Inf)
    prob = IntegralProblem(f, domain, param)
    sol = solve(prob, QuadGKJL())
    2π * sol.u
end

four_mode_overlap(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1) ≈ 2 / √π
##
p1s = 0:10
p2s = 0:10
ws = [√3 1; 1/√3 1/√5]
titles = ["w=√3"  "w=1"; "w=1/√3"  "w=1/√5"]

overlaps = [four_mode_overlap(p1, p2, 0, 0, 1, 1, 1, 1, w, w, 1, 1) for p1 in p1s, p2 in p2s, w in ws]
colorrange = (0, maximum(overlaps))
colormap = :viridis

with_theme(theme_latexfonts()) do
    fig = Figure(size=(800, 600))
    for n ∈ axes(overlaps, 4), m ∈ axes(overlaps, 3)
        ax = Axis(fig[m, n], xlabel="p1", ylabel="p2", title=titles[m, n])
        heatmap!(ax, p1s, p2s, overlaps[:, :, m, n]; colormap, colorrange)
    end
    Colorbar(fig[:, end + 1]; colorrange, colormap)
    save("Plots/covariance_entries.png", fig)
    fig
end