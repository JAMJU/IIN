function [value] = laplacian_value(i, j, window_size, nrow, ncol,  img, regul)
%Compute laplacian value between pixel i and j
value = 0.;
[i0, i1] = ind2sub([nrow, ncol], i);
[j0, j1] = ind2sub([nrow, ncol], j);
krock = 0.;
if i==j
    krock = 1.;
end

if abs(i0-j0) < window_size && abs(i1 - j1) < window_size
    colori = reshape(img(i0,i1, :), 3, 1);
    colorj = reshape(img(j0,j1, :), 3, 1);
    nbPixels = double(window_size^2);
    minx = max([1, i1 - window_size + 1, j1 - window_size + 1]);
    maxx = min([ncol - window_size + 1, i1, j1]);
    miny = max([1, i0 - window_size + 1, j0 - window_size + 1]);
    maxy = min([nrow - window_size + 1, i0, j0]);
    value = 0.;
    for x=minx:maxx
        for y=miny:maxy
            [mu, sigma] = get_mu_sigma([y,x], window_size, img);
            value = value + krock - (1./nbPixels)*(1. + ((colori - mu).')/(sigma + (regul/nbPixels)*eye(3))*(colorj - mu));
        end
    end
    
end
end

