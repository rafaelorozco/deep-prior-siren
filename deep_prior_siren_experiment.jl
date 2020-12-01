using PyPlot
using LinearAlgebra
using Statistics, Random, StatsBase, Distributions
using Images
using TestImages, Colors
using Flux
import Flux.Optimise.update!

include("cnn_siren.jl")


img = testimage("lighthouse")
imgg = Gray.(img)
mat = convert(Array{Float64}, imgg)

model_size_rows = 256
model_size_cols = 256
phantom = imresize(mat[1:512,1:512], (model_size_rows, model_size_cols))
phantom = convert(Array{Float32,2}, phantom)

#normalize 
vmin_d=0
vmax_d=255

phantom_range = maximum(phantom) - minimum(phantom)
phantom = (phantom .- minimum(phantom)) ./ phantom_range 
#phantom = vmax_d .* phantom

#set seed for same results
Random.seed!(123);

#Add noise 
mean_e = 0
sigma_e = std(phantom) #make sure this is right
e = rand(Normal(mean_e, sigma_e), size(phantom));
phantom_noise = Float32.(phantom + e);

#imshow(phantom_noise)
#colorbar()

################# Find prior distribution ###########################
phantom_true = Flux.unsqueeze(Flux.unsqueeze(phantom, 3), 3);

#use the paper sanctioned distribution and range
Random.seed!(123);
z = Float32.(rand(Uniform(0,0.1), size(phantom_true)[1], size(phantom_true)[2], 32, 1));

net_func, layers = CNN();

G(z) = net_func(z, layers);
w = Flux.params(layers);


#make "degraded" data
y = phantom_noise;

maxiter = 100
losses = zeros(maxiter);

#opt = Descent(0.00000001) #now only update weights #change to map in cost function
opt = ADAM(1f-4);

@time begin
for i in 1:maxiter

    loss, back = Flux.pullback(w) do
        Float32(1/(2*sigma_e^2.0))*(sum(G(z) - y).^2.0f0) #mle
       
    end
    grads = back(1f0)

    for p in w
        update!(opt, p, grads[p])
    end

    global losses[i] = loss
    
    print("Iteration: ", i[], "; f = ", loss, "\n")

end
end

figure();
imshow(G(z)[:,:,1,1]);title("predicted image");
figure();
imshow(G(z)[:,:,1,1]-y);title("predicted error");
figure();
imshow(y);title("noisy image");
figure();imshow(phantom);title("clean image");
figure();imshow(phantom-y);title("noise");

figure();plot(losses);title("loss history");