clear;
clc;

addpath(genpath('funs/'));
addpath(genpath('tSVD/'));

res = [];
[res_IMG]=[];
[res_PVC]=[];
[res_GPVCO]=[];

%% Load ORL dataset
f=1;
c = length(unique(truth));



load('dataset\ORL_4views.mat'); c=40;truth=truth';
load('dataset\ORL_e2_fold.mat');
ind_folds = folds{f};


numClust = length(unique(truth));
num_view =size(X,2);
[numFold,numInst]=size(ind_folds);
%fid=fopen('ORL_PER_10.txt','a');

result=[];
Y = cell(1,num_view);
for iv = 1:num_view
    X1 = X{iv}';
    X1 = NormalizeFea(X1,1);
    ind_0 = find(ind_folds(:,iv) == 0);
    X1(ind_0,:) = [];
    Y{iv} = X1';
    W1 = eye(size(ind_folds,1));
    W1(ind_0,:) = [];
    G{iv} = W1;
end
%% Graph construction
S_temp=graph_construction(Y);
for i=1:num_view
    S(:,:,i)=G{i}'*S_temp{i}*G{i};
    [nu,~]=size(S_temp{i});
    omega(:,:,i)=G{i}'*ones(nu,nu)*G{i};
end

%rlist=[1.25,1.26,1.29,1.3,1.35,1.4,1.50,20];
rlist = [1.25];

%plist = [0.2,0.3,0.4,0.5,0.6,0.7,0.8];
plist= [0.8];
%plist = [0.2,0.4,0.5,0.7];
%betalist=[0.1,0.15,0.3,0.35,0.9];
betalist= [0.9];
%betalist = [0.1 0.2 0.3];
%lambda1list =[0.5,0.35,5,10,100];
%lambda1list=[1.38,1.39,1.9,2.4,2.5];
%lambda1list = [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0];
%lambda1list = [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0];
lambda1list = [1.0,1.3,1.4,1.5,1.6,2.1,2.0,2.10,2.11,2.12,2.13,2.14,2.15,2.15,2.17,2.18,35,45,21,26];
%lambda2list = [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0];
lambda2list = [1.0,1.1,1.2,1.3,1.6,1.7,1.65,1.66,100,50,24,36,78,21];


maxIterTimes = 110;
num_of_samples = length(truth);
%% Training
allmaxValue = [];
c=40;
for rIndex = 1:length(rlist)
    r = rlist(rIndex);
    for pIndex = 1:length(plist)
        p = plist(pIndex);
        for betaIndex = 1:length(betalist)
            beta = betalist(betaIndex);
            for lambda1Index =1:length(lambda1list)
                for lambda2Index = 1:length(lambda2list)
                    lambda1 = lambda1list(lambda1Index);
                    lambda2 = lambda2list(lambda2Index);
                    mode=2;
                    [res] = My_comple(S_temp,G,truth,c,omega,beta,p,mode,r,0,lambda1,maxIterTimes,num_of_samples,lambda2,num_view);
                    tempR = r*ones(maxIterTimes,1); %这里的和里面的循环次数有关
                    tempP = p*ones(maxIterTimes,1);
                    tempBeta = beta*ones(maxIterTimes,1);
                    tempLambda1 = lambda1*ones(maxIterTimes,1);
                    res1 = [res,tempR,tempP,tempBeta,tempLambda1];
                    [result]=[result;res1];
                    fprintf(fid,'r: %f ',r);
                    fprintf(fid,'p: %f ',p);
                    fprintf(fid,'beta: %f ',beta);
                    fprintf(fid,'lambda2: %f \n',lambda2);
                    fprintf(fid,'lambda1: %f \n ',lambda1);
                    
                    for i = 1:110
                        fprintf(fid,'%g  %g %g  \n',res(i,1:3));
                    end
                    [temp,~] = max(res(:,1:3),[],1);
                    allmaxValue =[allmaxValue;temp];
                end
            end
        end
    end
end

save('file_new1.mat','allmaxValue')
%最好的acc
%  78.7000   86.3080   80.4000    1.2500    0.8000    0.3250    0.3250