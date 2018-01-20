function [mini] = min_patch(x0, x1, size_patch, A,  nrow, ncol, img)
% Return the minimum on a patch
half = int32(size_patch/2);
startx0 = x0 - half;
startx1 = x1 - half;
minR = 255;
minG = 255;
minB = 255;
for i=startx0:startx0+size_patch
    for j=startx1:startx1+size_patch
        if (i>=1 && j>=1 && i<=nrow && j<=ncol)
            R = img(i,j,1);
            G = img(i,j,2);
            B = img(i,j,3);
            if minR > R
                minR = R;
            end
            if minG > G
                minG = G;
            end
            if minB > B
                minB = B;
            end
            
        end
    end
end
Values = zeros([1,3]);
Values(1) = double(minR)/double(A(1));
Values(2) = double(minG)/double(A(2));
Values(3) = double(minB)/double(A(3));
mini = min(Values);
end

