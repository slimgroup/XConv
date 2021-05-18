using LinearAlgebra, PyPlot, Random, FFTW, Distributions, CircularArrays, Printf

close(:all)

ci, co, N, ps, b = 16, 16, 32*32, 64, 10

eb = sign.(randn(Float32, co*N, ps))

chunk(arr, n) = [arr[i:min(i + n - 1, end)] for i in 1:n:length(arr)]
simil(x, y) = 100*dot(x[:], y[:])/(norm(x)*norm(y))
error(x, y) = norm(x - y)

function draw_e(ps::Integer, co::Integer, N::Integer)
    n = ps ÷ co
    if n < 8
        n = 8
        inds = chunk(randperm(n*co) .% ps .+ 1, n)
    else
        inds = chunk(randperm(ps), n)
    end
    e = zeros(Float32, N*co, ps)
    for i=1:co
        # iloc = inds[(i-1)*n+1:i*n]
        iloc = inds[i]
        ni = length(iloc)
        e[(i-1)*N+1:i*N, iloc] .= Float32(sqrt(ps/(ni)))*sign.(randn(Float32, N, ni))
    end
    e = e
    return e, collect(e')
end

eortho, eorthot = draw_e(ps, co, N)

figure()
subplot(221)
imshow(eb, vmin=-ps/10, vmax=ps/10, cmap="seismic", aspect="auto")
title("Z, Random")
subplot(222)
imshow(eortho, vmin=-ps/10, vmax=ps/10, cmap="seismic", aspect="auto")
title("Z, Random Orthogonalized")
subplot(223)
imshow(eb*eb', vmin=-ps/10, vmax=ps/10, cmap="seismic", aspect="auto")
title(L"$ZZ^T$ Random")
subplot(224)
imshow(eortho*eorthot, vmin=-ps/10, vmax=ps/10, cmap="seismic", aspect="auto")
title(L"$ZZ^T$ Random Orthogonalized")
tight_layout()

####### PROBE 3D Tensore with it
probe_trace(ci::Integer, co::Integer, N::Integer, A::Matrix{T}, e::Matrix{T}, et) where T = probe_trace(ci, co, N, A, e, collect(et))

function probe_trace(ci::Integer, co::Integer, N::Integer, A::Matrix{T}, e::Matrix{T}, et::Matrix{T}=e') where T
    # A * z
    xyz = A * e
    trl = zeros(T, ci, co)
    for cii = 1:ci
        for coi = 1:co
            eloc = view(et, :, (coi - 1)*N+1:coi*N)
            xyxloc = view(xyz, (cii - 1)*N+1:cii*N, :)
            trl[cii, coi] = LinearAlgebra.tr(eloc * xyxloc)/ps
        end
    end
    return trl
end

A = vcat([hcat([rand(-100:100) * randn(Float32, N, N) for i=1:co]...) for j=1:ci]...)
block(A, i, j) = A[(i-1)*N+1:i*N, (j-1)*N+1:j*N]

true_tr = zeros(Float32, ci, co)
[true_tr[i, j] = LinearAlgebra.tr(block(A, i, j)) for i=1:ci for j=1:co]

# Single sample
trr = probe_trace(ci, co, N, A, eb)
tro = probe_trace(ci, co, N, A, eortho, eorthot)

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
# me = eb*eb'
# meo = eortho*eorthot

# Mean
np = 200
convergence = zeros(Float32, np, 2)
similarity = zeros(Float32, np, 2)
convergence[1, 1] = error(true_tr, trr)
convergence[1, 2] = error(true_tr, tro)

similarity[1, 1] = simil(true_tr, trr)
similarity[1, 2] = simil(true_tr, tro)
for i=2:np
    local eeb = randn(Float32, co*N, ps)
    local eeortho, eeorthot = draw_e(ps, co, N)
    # global me = ((i-1)*me + eeb*eeb')/i
    # global meo = ((i-1)*meo + eeortho*eeorthot)/i

    global trr = ((i-1)*trr + probe_trace(ci, co, N, A, eeb))/i
    global tro = ((i-1)*tro + probe_trace(ci, co, N, A, eeortho, eeorthot))/i
    convergence[i, 1] = error(true_tr, trr)
    convergence[i, 2] = error(true_tr, tro)
    similarity[i, 1] = simil(true_tr, trr)
    similarity[i, 2] = simil(true_tr, tro)
    @printf("%d/%d : Er = %2.2e, Sr = %2.2f, Eo = %2.2e, So = %2.2f \n",i, np,
            convergence[i, 1], similarity[i, 1], convergence[i, 2], similarity[i, 2])
end

ni(x) = x / norm(x, Inf)

# figure()
# subplot(121)
# imshow(me, vmin=-ps/10, vmax=ps/10, cmap="seismic")
# title(L"Mean Random $ZZ^T$")
# subplot(122)
# imshow(meo, vmin=-ps/10, vmax=ps/10, cmap="seismic")
# title(L"Mean Random Orthogonalized $ZZ^T$")

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
semilogx(similarity[1:np, 1], label="Similarity random")
semilogx(similarity[1:np, 2], label="Similarity random ortho")
legend()


# independent
# r x ci x batch
# all
# r x batch