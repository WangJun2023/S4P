function [labels,img, imgMulti]= cubseg(indian_pines,superpixel_num,p2)


[M,N,B]=size(indian_pines);
Y_scale=scaleForSVM(reshape(indian_pines,M*N,B));
% Y_scale = reshape(indian_pines,M*N,B);
Y=reshape(Y_scale,M,N,B);
p = 1;

[Y_pca] = pca(Y_scale, p);
[Y_pcaMulti] = pca(Y_scale, p2);
% Y_pca = mean(Y_scale,2);
img = im2uint8(mat2gray(reshape(Y_pca', M, N, p)));
imgMulti = im2uint8(mat2gray(reshape(Y_pcaMulti', M, N, p2)));

[Ratio]=Edge_ratio3(img);

sigma=0.05;
% K=cc;
% K=SegPara*Ratio;
grey_img = im2uint8(mat2gray(Y(:,:,30)));
labels = mex_ers(double(img),superpixel_num);
[height width] = size(grey_img);
[bmap] = seg2bmap(labels,width,height);
bmapOnImg = img;
idx = find(bmap>0);
timg = grey_img;
timg(idx) = 255;
bmapOnImg(:,:,2) = timg;
bmapOnImg(:,:,1) = grey_img;
bmapOnImg(:,:,3) = grey_img;

% figure;
% imshow(bmapOnImg,[]);
% imwrite(grey_img,'bmapOnImg.bmp')
% title('superpixel boundary map');

end

function [Ratio]=Edge_ratio3(img)
 [m,n] = size(img);
%  img =  rgb2gray(img);
 BW = edge(img,'log');
%  figure,imshow(BW)
 ind = find(BW~=0);
 Len = length(ind);
 Ratio = Len/(m*n);
end