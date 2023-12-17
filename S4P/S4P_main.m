function [OA,MA,Kappa] = S4P_main(Dataset,classifier_type,SuperPixelNo)

img_src = Dataset.A;
[W, H, L]=size(img_src);

V = 3;
[labels, ~, ImgMulti]= cubseg(img_src,SuperPixelNo,V);

S = zeros(SuperPixelNo,SuperPixelNo);
TotalS = S;

%% Initial Sv
options.t=1;
options.bSelfConnected = 1;
options.NeighborMode = 'KNN';
options.WeightMode = 'HeatKernel';
options.k = 0;

X = zeros(L,SuperPixelNo);
labels = labels + 1;
idx = label2idx(labels);
X1 = (Dataset.X)';
[m1,n1]=size(X1);
for i = 1 : SuperPixelNo
    [r,c] = ind2sub([m1,n1],idx{i});
    X(:,i) = (mean(X1(r,:)))';
end
X = X./ repmat(sqrt(sum(X.^2, 2)),1,SuperPixelNo);

[W_1,H_1,L_1]=size(ImgMulti);
TempImg = reshape(ImgMulti,W_1*H_1,L_1);
[m2,n2]=size(TempImg);
for i = 1 : SuperPixelNo
    [r,c]=ind2sub([m2,n2],idx{i});
    TempFea(:,i) = (mean(TempImg(r,:)))';
end
TempFea = TempFea./ repmat(sqrt(sum(TempFea.^2, 2)),1,SuperPixelNo);

for vIndex = 1:V
    disp(['PCA: ',num2str(vIndex)]);
    tempS = constructW(TempFea(vIndex,:)',options);
    for si = 1:SuperPixelNo
        tempS(si,:) = tempS(si,:)./sum(tempS(si,:));
    end
    S_d.data{vIndex}= (tempS + tempS')/2;
    TotalS = TotalS + S_d.data{vIndex};
end

%% Initial S
TotalS = TotalS/V;
for TotalSi = 1:SuperPixelNo
    TotalS(TotalSi,:) = TotalS(TotalSi,:)./sum(TotalS(TotalSi,:));
end
S = (TotalS+TotalS')./2;

gammaSet = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000];
alphaSet = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000];
BandK = 5 : 5 : 50;
count = 1;
IE = Entrop(Dataset.X);

for alphaIndex = 1 : length(alphaSet)
    for gammaIndex = 1 : length(gammaSet)
        maxIter = 10;
        
        [Z, S, Wo, obj] = S4P(X', alphaSet(alphaIndex), gammaSet(gammaIndex), maxIter, S_d, S, V);
        for iBand = 1 : length(BandK)
            K = BandK(iBand);
            disp(['Band is ',num2str(K)]);
            Y = predict(Z,K);
            [acc,Classify_map] = test_bs_accu(Y, Dataset, classifier_type);
            OA(count,iBand) = acc.OA;
            MA(count,iBand) = acc.MA;
            Kappa(count,iBand) = acc.Kappa;
            STDOA(count,iBand) = acc.STDOA;
            STDMA(count,iBand) = acc.STDMA;
            STDKappa(count,iBand) = acc.STDKappa;
        end
        count = count + 1;
    end
end

OA = max(OA);
MA = max(MA);
Kappa = max(Kappa);
