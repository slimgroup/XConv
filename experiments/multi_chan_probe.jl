using LinearAlgebra, PyPlot, Random, FFTW, Distributions, CircularArrays, Printf, Statistics

close(:all)

ci, co, N, ps, b = 16, 16, 256, 64, 50
plot_I = false

eb = randn(Float32, co*N, ps)

chunk(arr, n) = [arr[i:min(i + n - 1, end)] for i in 1:n:length(arr)]
simil(x, y) = 100*dot(x[:], y[:])/(norm(x)*norm(y))
error(x, y) = norm(x - y)/(norm(x)+norm(y))

function draw_e(ps::Integer, co::Integer, N::Integer)
    n = ps ÷ co
    if n < 4
        n = 4
        inds = chunk(randperm(n*co) .% ps .+ 1, n)
        overlap = n / (ps ÷ co)
    else
        inds = chunk(randperm(ps), n)
        overlap = 1
    end
    e = zeros(Float32, N*co, ps)
    for i=1:co
        iloc = inds[i]
        ni = length(iloc)
        e[(i-1)*N+1:i*N, iloc] .= Float32(sqrt(ps/(overlap*ni)))*randn(Float32, N, ni)
    end

    return e
end

eortho = draw_e(ps, co, N)

if plot_I
    plot_e_scale = ps/50
    figure()
    subplot(221)
    imshow(eb, vmin=-.1, vmax=.1, cmap="seismic", aspect="auto")
    title("Z, Random")
    subplot(222)
    imshow(eortho, vmin=-.1, vmax=.1, cmap="seismic", aspect="auto")
    title("Z, Random Orthogonalized")
    subplot(223)
    imshow(eb*eb', vmin=-plot_e_scale, vmax=plot_e_scale, cmap="seismic", aspect="auto")
    title(L"$ZZ^T$ Random")
    subplot(224)
    imshow(eortho*eortho', vmin=-plot_e_scale, vmax=plot_e_scale, cmap="seismic", aspect="auto")
    title(L"$ZZ^T$ Random Orthogonalized")
    tight_layout()
    savefig("./figures/zortho.png", bbox_inches="tight")
end

####### PROBE 3D Tensore with it

# Simple X Y^T struct to avoid blowing up memory
struct outer_LR
    L
    R
end

Base.:*(A::outer_LR, x) = A.L*(A.R'*x)
Base.:*(x, A::outer_LR) = (x*A.L)*A.R'
block(A::outer_LR, i, j) = view(A.L, (i-1)*N+1:i*N, :) * view(A.R, (j-1)*N+1:j*N, :)'

a = vcat([rand(-5:5) * rand(Float32, N, b) for i=1:ci]...)

A = outer_LR(a, max.(0f0, 100 .* randn(Float32, N*co, b)))


function probe_trace(ci::Integer, co::Integer, N::Integer, A::outer_LR, e::Matrix{T}) where T
    # A * z
    xyz = A * e
    trl = zeros(T, ci, co)
    for cii = 1:ci
        for coi = 1:co
            eloc = view(e, (coi - 1)*N+1:coi*N, :)'
            xyxloc = view(xyz, (cii - 1)*N+1:cii*N, :)
            trl[cii, coi] = LinearAlgebra.tr(eloc * xyxloc)/ps
        end
    end
    return trl
end
true_tr = zeros(Float32, ci, co)
[true_tr[i, j] = LinearAlgebra.tr(block(A, i, j)) for i=1:ci for j=1:co]

# Single sample
trr = probe_trace(ci, co, N, A, eb)
tro = probe_trace(ci, co, N, A, eortho)

function plot_tr(trr, tro, true_tr)
    iplot = rand(1:ci, 4)
    figure()
    for (ip, i)=enumerate(iplot)
        subplot(2, 2, ip)
        title("Input channel $(i)")
        plot(trr[i, :], "--*r", label="random", linewidth=1, markersize=5)
        plot(tro[i, :], "--ob", label="ortho random", linewidth=1, markersize=5, markerfacecolor="none")
        plot(true_tr[i, :], "--xk", label="true", linewidth=1, markersize=5)
    end
    legend()
end

plot_tr(trr, tro, true_tr)

# Mean
np = 5000

convergence = zeros(Float32, np, 2)
similarity = zeros(Float32, np, 2)
convergence[1, 1] = error(true_tr, trr)
convergence[1, 2] = error(true_tr, tro)

similarity[1, 1] = simil(true_tr, trr)
similarity[1, 2] = simil(true_tr, tro)

i=1

@printf("%d/%d : Er = %2.2e, Sr = %2.2f, Eo = %2.2e, So = %2.2f \n",i, np,
        convergence[i, 1], similarity[i, 1], convergence[i, 2], similarity[i, 2])

for i=2:np
    local eeb = randn(Float32, co*N, ps)
    local eeortho = draw_e(ps, co, N)

    global trr = ((i-1)*trr + probe_trace(ci, co, N, A, eeb))/i
    global tro = ((i-1)*tro + probe_trace(ci, co, N, A, eeortho))/i
    convergence[i, 1] = error(true_tr, trr)
    convergence[i, 2] = error(true_tr, tro)
    similarity[i, 1] = simil(true_tr, trr)
    similarity[i, 2] = simil(true_tr, tro)
    @printf("%d/%d : Er = %2.2e, Sr = %2.2f, Eo = %2.2e, So = %2.2f \n",i, np,
            convergence[i, 1], similarity[i, 1], convergence[i, 2], similarity[i, 2])
end

ni(x) = x / norm(x, Inf)

plot_tr(trr, tro, true_tr)

figure()
plot((true_tr[:] .- trr[:])./true_tr[:], label="Relative error random")
plot((true_tr[:] .- tro[:])./true_tr[:], label=" Relative error ortho random")
legend()

figure()
subplot(211)
loglog(convergence[1:np, 1], label="Error random")
loglog(convergence[1:np, 2], label="Error random ortho")
legend()
subplot(212)
plot(similarity[1:np, 1], label="Similarity random")
plot(similarity[1:np, 2], label="Similarity random ortho")
yticks([i*10 for i=0:2:10], labels=["$(i*10)%" for i=0:2:10])
legend()


# independent
# r x ci x batch
# all
# r x batch