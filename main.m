% Parameters
size_patch = 5;
size_image = 160;
w= 0.95;
lamb = 0.1;
window_size = 2;
eps = 0.0001;

% Get the image and resize it
imrgb = imread('Data/image1.jpeg');
[nrow,ncol,nchan] = size(imrgb);
%figure;
%subplot(2,2,1); imshow(imrgb,[]), title('color image')
%subplot(2,2,2), imshow(imrgb(:,:,1),[]), title('red channel');
%subplot(2,2,3), imshow(imrgb(:,:,2),[]), title('green channel');
%subplot(2,2,4), imshow(imrgb(:,:,3),[]), title('blue channel');
im = imresize(imrgb,[NaN, size_image]);
%figure;
%imshow(im);
[nrows, ncols, nchans]= size(im);
im2 = im;
im = double(im)/255.;
figure;
imshow(im);
% Get grayscale
im_gray = rgb2gray(im2);

% Create dark channel
dark_channel = double(zeros([nrows, ncols]));
for i=1:nrows
    for j=1:ncols
        dark_channel(i,j) = min_patch(i,j,size_patch,[1.,1.,1.],nrows,ncols,im);
    end
end
figure;
imshow(dark_channel);

% recover A
% first get 1 percent brightest in the dark channel
percent = int32(nrows*ncols/100);
[sortedIm, Indexes] = sort(dark_channel(:), 'descend');
percent_selected = Indexes(1:percent);
[value, ind] = max(im_gray(percent_selected));
A = zeros([1, 3]);
% We only choose one for the moment
[i,j] = ind2sub(size(dark_channel),percent_selected(1));
A(1) = im(i,j, 1);
A(2) = im(i,j, 2);
A(3) = im(i,j, 3);
disp('A recovered');
disp(A);

% recover constant t_tild
t_tild = double(zeros([nrows, ncols]));
for i=1:nrows
    for j=1:ncols
         t_tild(i,j) = 1. - w*min_patch(i,j,size_patch, A, nrows, ncols, im);
    end
end
figure;
imshow(t_tild);

t = t_tild;

% We recover J
t0 = 0.1;
J = zeros([nrows, ncols, 3]);
for i= 1:nrows
    for j = 1:ncols
        J(i,j,1) = (im(i,j,1) - A(1))/max(t(i,j),t0) + A(1);
        J(i,j,2) = (im(i,j,2) - A(2))/max(t(i,j),t0) + A(2);
        J(i,j,3) = (im(i,j,3) - A(3))/max(t(i,j),t0) + A(3);
    end
end
%J2 = uint8(255*J);
%disp(J(:,1))
figure;
imshow(J);

% Soft matting
% Creation of the laplacian matrix
L = double(zeros([nrows*ncols,nrows*ncols]));
test = 0;
for i=1:nrows*ncols
    for j= 1:nrows*ncols
        L(i,j) = laplacian_value(i,j,window_size,nrows,ncols,im,eps);
        if L(i,j) ~=0
            test = test +1;
        end
    end
end
disp('End');
disp(test);

% We use the preconditined gradient method
t = pcg(L + lamb*double(eye(size(L))), lamb*reshape(t_tild, [], 1), 1e-5, 1000);
t = reshape(t, [nrows, ncols]);
%disp(t(:,1))
%disp(t(:,2))
figure;
imshow(t);
% We recover J
t0 = 0.1;
J = zeros([nrows, ncols, 3]);
for i= 1:nrows
    for j = 1:ncols
        J(i,j,1) = (im(i,j,1) - A(1))/max(t(i,j),t0) + A(1);
        J(i,j,2) = (im(i,j,2) - A(2))/max(t(i,j),t0) + A(2);
        J(i,j,3) = (im(i,j,3) - A(3))/max(t(i,j),t0) + A(3);
    end
end
%J2 = uint8(255*J);
%disp(J(:,1))
figure;
imshow(J);







