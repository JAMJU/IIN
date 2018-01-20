function [mu,sigma] = get_mu_sigma(window, window_size, img)
%Return the mean and the covariance matrix for a certain window on img
wind = img(window(1):window(1) + window_size - 1, window(2):window(2) + window_size - 1, :);
R = double(wind(:,:,1));
G = double(wind(:,:,2));
B = double(wind(:,:,3));
mu = zeros([3,1]);
sigma = zeros([3,3]);
RG = cov(R,G);
GB = cov(G,B);
RB = cov(R,B);
sigma(1,1) = RG(1,1);
sigma(2,2) = RG(2,2);
sigma(3,3) = GB(2,2);
sigma(1,2) = RG(1,2);
sigma(2,1) = RG(2,1);
sigma(2,3) = GB(1,2);
sigma(3,2) = GB(2,1);
sigma(1,3) = RB(1,2);
sigma(3,1) = RB(2,1);
mu(1) = mean(R(:));
mu(2) = mean(G(:));
mu(3) = mean(B(:));
end

