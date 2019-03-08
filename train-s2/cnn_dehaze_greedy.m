function [net, info] = cnn_dehaze_greedy()

clear;clc;close all;

opts.expDir = fullfile(vl_rootnn, 'train-s2', 'output-g');
opts.imdbPath = './datasets/train/';

opts.train.continue = true;
opts.train.gpus = 1;
opts.train.expDir = opts.expDir;
opts.train.stats = {'lossJ', 'lossT', 'lossU'};
opts.train.train = 1:18000;
opts.train.val   = 18001:20000;
opts.train.solver = @adam;
opts.train.batchSize = 25;
opts.train.numEpochs = 20;
opts.train.learningRate = 1e-3;

% network initialization
net = cnn_dehaze_init_greedy();

% training process
[net, info] = cnn_dehaze_train(net, opts.imdbPath, @getBatch, opts.train);

end

function inputs = getBatch(imdb, batch)

bs = length(batch);
noisy = zeros(240, 240, 3, bs, 'single');
clean = zeros(240, 240, 3, bs, 'single');
ttmap = zeros(240, 240, 1, bs, 'single');
parfor i=1:bs
    data = load(strcat(imdb, sprintf('train-%04d.mat', batch(i))));
    light = airLight(data.noisy);
    for c=1:3
        noisy(:,:,c,i) = data.noisy(:,:,c) / light(c);
        clean(:,:,c,i) = data.clean(:,:,c) / light(c);
    end
    ttmap(:,:,:,i) = data.ttmap;
end
noisy = gpuArray(noisy);
clean = gpuArray(clean);
darkc = dy_dark_channel(clean, 7);
inputs = {'noisy', noisy, 'clean', clean, 'ttmap', ttmap, 'darkc', darkc};

end

function light = airLight(image)

image = gpuArray(image);
darkc = dy_dark_channel(image, 15);
[~, index] = sort(gather(darkc(:)), 'descend');
index = index(floor(0.001*numel(darkc)));

light = zeros(3,1);
image = reshape(gather(image), [], 3);
for c=1:3
    tpimg = image(:, c);
    light(c) = tpimg(index);
end

end