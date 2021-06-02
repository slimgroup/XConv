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
    if n < 8
        n = 8
        inds = chunk(randperm(n*co) .% ps .+ 1, n)
        overlap = 1 #n / (ps ÷ co)
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

