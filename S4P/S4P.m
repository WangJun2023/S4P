function [Z, S, W,obj] = S4P(X, alpha, gamma, maxIter, S_d, S, group_num)

[n, d] = size(X);

%% Initialize input parameters
W = ones(group_num, n)./group_num;
[Z, ~] = InitializeSIGs(X);

obj = zeros(maxIter, 1);
w = ones(1,group_num)/group_num;
islocal = 1; % only update the similarities of neighbors if islocal=1
lambda = 1;

for iter = 1:maxIter
    disp(['Alpha, Gamma: ',num2str(alpha),', ',num2str(gamma),', Iter: ',num2str(iter)]);
    
    %% update w
    for v = 1:group_num
        US = S - S_d.data{v};
        distUS = norm(US, 'fro')^2;
        if distUS == 0
            distUS = eps;
        end
        w(v) = 0.5/sqrt(distUS);
    end
    
    
    %% update S
    disp('update S...');
    
    % update U
    F = X*Z;
    dist = L2_distance_1(F',F');
    S = zeros(n);
    for i=1:n
        idx = zeros();
        for v = 1:group_num
            s0 = S_d.data{v}(i,:);
            idx = [idx,find(s0>0)];
        end
        idxs = unique(idx(2:end));
        if islocal == 1
            idxs0 = idxs;
        else
            idxs0 = 1:n;
        end
        for v = 1:group_num
            s1 = S_d.data{v}(i,:);
            si = s1(idxs0);
            di = dist(i,idxs0);
            mw = group_num*w(v);
            lmw = lambda/mw;
            q(v,:) = si-0.5*lmw*di;
        end
        S(i,idxs0) = SloutionToP19(q,group_num);
        clear q;
    end

    for si = 1:n
        S(si,:) = S(si,:)./sum(S(si,:));
    end
    S = S - diag(diag(S));
    
    clear B_i a_i part_m psi temp;
    
    %% update Z
    disp('update Z...');
    LapMatrix = diag(sum(S, 1)) - S;
    for loop = 1 : maxIter
        temp = 2 * sqrt(sum(Z.^2, 2)) + 1e-15;
        Q = diag(1./temp);
        temp1 = X' * (eye(n) + alpha * LapMatrix) * X + gamma * Q;
        Z = temp1 \ X' * X;
    end
    Z = Z - diag(diag(Z));
    Z = max(Z,eps);
    clear temp Q temp1;
    
    %% calculate objective function value1
    disp('calculate obj-value...');
    
    temp_S_i = 0;
    for v = 1:group_num
        temp_S_i = temp_S_i + w(v)*norm(S-S_d.data{v},'fro')^2;
    end
    temp_formulation1 = temp_S_i;
    temp_formulation2 = norm(X-X*Z,'fro')^2;
    temp_formulation3 = gamma * sum(sqrt(sum(Z.^2, 2)));%L2,1-norm
    temp_formulation4 = alpha * trace(Z'*X'*LapMatrix * X * Z);
    obj(iter) = temp_formulation1 + temp_formulation2 + temp_formulation3 + temp_formulation4;
    if (iter>2) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-6 || iter>maxIter)
        break;
    end
    clear temp_formulation1 temp_S_j temp_S_i temp_formulation2 temp_formulation3 temp_formulation4;
    
end

end