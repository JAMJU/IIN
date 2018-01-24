
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