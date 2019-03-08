clear;clc;close all;

image = imread('haze-02.jpg');
image = imresize(image, 1);

[resim, restt] = ours_tiphqs_s1_eval(image);

figure, imshow(image)
figure, imshow(resim)
figure, imshow(restt)