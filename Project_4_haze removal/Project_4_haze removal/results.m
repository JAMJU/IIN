% Show results
function results(image) 

im = imread(image);
% normalize
%im=imresize(im,.5);
im = double(im);
im = im./255;
J = deHaze(im);
figure;
subplot(1,2,1);
imagesc(im)
title 'Original'
axis image off;
subplot(1,2,2);
imagesc(J)
title 'De-hazed'
axis image off;