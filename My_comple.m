function res=My_comple(S,G,truth,c,omega,b,p,mode,r,lambda,lambda1,maxIterTimes,num_of_samples,lambda2)
S_inner = S;
Q2 = cell(1,2);
P = cell(1,2);
EE =  zeros(num_of_samples,num_of_samples);
T = zeros(num_of_samples,num_of_samples);
I1 = zeros(num_of_samples,num_of_samples);
HH = zeros(num_of_samples,c);
newQ1 = zeros(num_of_samples,c);


%% Initialization
Wv = [1/2 1/2]
[~,n2,~]=size(S_inner);

for i=1:n2
    S_sss(:,:,i)=G{i}'*S_inner{i}*G{i};
    Q2{i}=zeros(size(S_inner{i}));
    P{i}=Q2{i};
    E{i}=P{i};
end

FF =  cell(1,2);
for i = 1:2
     FF{i}=zeros(num_of_samples,c) 
end;

FF_old = FF;
dim=size(S_sss);
[n1,n2,n3]=size(S_sss);
mu=1e-4;
tol = 1e-8; max_iter = maxIterTimes; rho = 1.1;

Q1=zeros(dim); Q3=zeros(dim); W=zeros(dim); Y=zeros(dim);
Z=zeros(dim); M=zeros(dim); F=zeros(n2,c); U=zeros(n2,c);
iter=0;
n_v = 2;
temp_W = cell(1,n_v);
temp_W_LD =  cell(1,n_v);
alpha=1/n3*ones(1,n3);

Z=S_sss;
beta = [10,50]';
H = zeros(num_of_samples,c);
for i=1:n3
    [nu,~]=size(S_inner{i});
    omega(:,:,i)=G{i}'*ones(nu,nu)*G{i};%ones(numFold,numFold);
end

for i = 1:2
    alpha(i) =0.5
end;
%初始值Ls,以后每步计算都会进行迭代
Ls = zeros(num_of_samples,num_of_samples);
for i=1:max_iter
    iter=iter+1;
    X_k=Z;
    Z_k=M;
    for j=1:n3
        P{j}=S_inner{j}+1/mu*Q2{j}-E{j};
    end
    %% Update Z
    for j=1:n3
        Z(:,:,j)=0.5*(M(:,:,j)-1/mu*Q1(:,:,j)+W(:,:,j)-1/mu*Q3(:,:,j))-1/mu*(Y(:,:,j).*omega(:,:,j));
    end
    
    %% Update  B
    for j=1:n3
        temp1 = L2_distance_1(FF{j}',FF{j}');
        temp2 = Z(:,:,j)+Q3(:,:,j)/mu;
        linshi_W = temp2-alpha(j)^r*b*temp1/mu;
        linshi_W = linshi_W-diag(diag(linshi_W));
        for ic = 1:size(Z(:,:,j),2)
            ind = 1:size(Z(:,:,j),2);
            %             ind(ic) = [];
            W(ic,ind,j) = EProjSimplex_new(linshi_W(ic,ind));
        end
    end
    clear temp1 temp2
     z_xiangliang = Z(:);
     w = Q1(:);
     sX = size(Z);
     [g, ~] = wshrinkObj_weight(z_xiangliang + 1/rho*w,beta/rho,sX,0,3);
     M = reshape(g, sX);
    %% Update M
   % [M,~,~] = prox_tnn(Z+Q1/mu,beta/mu,p,mode);
    
    %% Update F

    n_v = size(W,3)

    for j=1:n3
        temp_W{j} = W(:,:,j);
        temp_W{j} = (temp_W{j}+temp_W{j}')/2;    
        L_D = diag(sum(temp_W{j}));
        temp_W_LD{j} = L_D - temp_W{j};
    end

  
    for kkk = 1:n3
          
        tempSum = 2*lambda1*temp_W_LD{kkk} - 2*Wv(kkk)*H*H' + Wv(kkk)*eye(num_of_samples);
        FF{kkk} = eig1(tempSum,c,0); 
    end
    
    %Update HH
    tempSum1 = zeros(num_of_samples,num_of_samples);
    for i = 1:2
        tempSum1 =tempSum1+ Wv(i)*(eye(num_of_samples)-2*FF{i}*FF{i}');
    end;
    tempSum1 = tempSum1 + lambda2*(eye(num_of_samples)-2*T');
    HH = eig1(tempSum1,c,0);
    
    
    
    %update T
    R = HH*HH';
    EE = L2_distance_1(newQ1',newQ1');
    linshi_W = R - EE/(2*lambda2);
    linshi_W = linshi_W-diag(diag(linshi_W));
    for rowIndexIndex = 1:num_of_samples  %row
        ind = 1:num_of_samples;
        T(rowIndexIndex,ind) = EProjSimplex_new(linshi_W(rowIndexIndex,ind));
    end 
    
    %updata new_Q1
    TT1 = (T+T')/2;
    D_T = diag(sum(TT1));
    L_T= D_T-TT1;
    
    [newQ1, ~, ~]=eig1(L_T, c, 0);
    
    
    %%
    
    %% Update E
    for j=1:n3
        temp1 = S_inner{j}-P{j}+Q2{j}/mu;
        temp2 = lambda/mu;
        E{j}= max(0,temp1-temp2)+min(0,temp1+temp2);
    end
    
    clear temp1 temp2
    %% Update alpha

    %% Checking Convergence
    chgX=max(abs(Z(:)-X_k(:)));
    chgZ=max(abs(M(:)-Z_k(:)));
    chgX_Z=max(abs(Z(:)-M(:)));
    chg=max([chgX chgZ chgX_Z]);
    
    if iter == 1 || mod(iter, 10) == 0
        disp(['iter ' num2str(iter) ', mu = ' num2str(mu) ', chg = ' num2str(chg) ', chgX = ' num2str(chgX) ', chgZ = ' num2str(chgZ) ',chgX_Z = ' num2str(chgX_Z) ]);
    end
    
    if chg<tol
        break;
    end
    %% Update Lagrange multiplier
    for j=1:n3
        tP(:,:,j)=G{j}'*P{j}*G{j};
    end
    Q1=Q1+mu*(Z-M);
    for j=1:n3
        Q2{j}=Q2{j}+mu*(S_inner{j}-P{j}-E{j});
    end
    Q3=Q3+mu*(Z-W);
    Y=Y+mu*(Z-tP).*omega;
    mu=rho*mu;

    %% Clustering
    new_F = newQ1;
  
    norm_mat = repmat(sqrt(sum(new_F.*new_F,2)),1,size(new_F,2));
    for i = 1:size(norm_mat,1)
        if (norm_mat(i,1)==0)
            norm_mat(i,:) = 1;
        end
    end
    new_F = new_F./norm_mat;
    Q = new_F;
    repeat = 5;
    for iter_c = 1:repeat
        pre_labels    = kmeans(real(Q),c,'emptyaction','singleton','replicates',20,'display','off');
        result_LatLRR = ClusteringMeasure(truth, pre_labels);
        AC(iter_c)    = result_LatLRR(1)*100;
        MIhat(iter_c) = result_LatLRR(2)*100;
        Purity(iter_c)= result_LatLRR(3)*100;
    end
    mean_ACC = mean(AC);
    mean_NMI = mean(MIhat);
    mean_PUR = mean(Purity);
    [res(iter,:)] =[mean_ACC,mean_NMI,mean_PUR];
    if iter == 1 || mod(iter, 10) == 0
        disp(['iter ' num2str(iter) ', mean_ACC = ' num2str(mean_ACC) ', mean_NMI = ' num2str(mean_NMI) ', mean_PUR = ' num2str(mean_PUR)]);
        disp(['---------------------------------------------------------------------------------------'])
    end
end
%  78.7000   86.3080   80.4000    1.2500    0.8000    0.3250    0.3250