%% Dongrui WU, drwu@hust.edu.cn

clc; clearvars; close all; %rng(0);

nMFs=2; % number of MFs in each input domain
alpha=.01; % initial learning rate
lambda=0.05; % L2 regularization coefficient
P=0.5; % DropRule rate
nIt=200; % number of iterations
Nbs=64; % batch size

temp=load('NO2.mat');
data=temp.data;
X=data(:,1:end-1); y=data(:,end); y=y-mean(y);
X = zscore(X); [N0,M]=size(X);
N=round(N0*.7);
idsTrain=datasample(1:N0,N,'replace',false);
XTrain=X(idsTrain,:); yTrain=y(idsTrain);
XTest=X; XTest(idsTrain,:)=[];
yTest=y; yTest(idsTrain)=[];
RMSEtrain=zeros(6,nIt); RMSEtest=RMSEtrain;

% Specify the total number of rules; use the original features without dimensionality reduction
nRules=30; % number of rules
[RMSEtrain(1,:),RMSEtest(1,:)]=MBGD_RDA2_G(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nRules,nIt,Nbs); % Gaussian MFs
[RMSEtrain(2,:),RMSEtest(2,:)]=MBGD_RDA2_T(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nRules,nIt,Nbs); % Trapezoidal MFs

% Dimensionality reduction
maxFeatures=5; % maximum number of features to use
if M>maxFeatures
    [~,XPCA,latent]=pca(X);
    realDim98=find(cumsum(latent)>=.98*sum(latent),1,'first');
    usedDim=min(maxFeatures,realDim98);
    X=XPCA(:,1:usedDim); [N0,M]=size(X);
end
XTrain=X(idsTrain,:); XTest=X; XTest(idsTrain,:)=[];

% Specify the number of MFs in each input domain
% Assume x1 has two MFs X1_1 and X1_2; then, all rules involving the first FS of x1 use the same X1_1, 
% and all rules involving the second FS of x1 use the same X1_2
[RMSEtrain(3,:),RMSEtest(3,:)]=MBGD_RDA_G(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nMFs,nIt,Nbs); % Gaussian MFs
[RMSEtrain(4,:),RMSEtest(4,:)]=MBGD_RDA_T(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nMFs,nIt,Nbs); % Trapezoidal MFs

% Specify the total number of rules; each rule uses different membership functions
nRules=nMFs^M; % number of rules
[RMSEtrain(5,:),RMSEtest(5,:)]=MBGD_RDA2_G(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nRules,nIt,Nbs); % Gaussian MFs
[RMSEtrain(6,:),RMSEtest(6,:)]=MBGD_RDA2_T(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nRules,nIt,Nbs); % Trapezoidal MFs

%% Plot results
figure('Position', get(0, 'Screensize')); hold on;
linestyles={'k--','k-','g--','g-','b--','b-','r--','r-','m--','m-','c--','c-'};
for i=1:size(RMSEtrain,1)
    plot(RMSEtrain(i,:),linestyles{2*i-1},'linewidth',1); 
    plot(RMSEtest(i,:),linestyles{2*i},'linewidth',2);
end
legend('Training RMSE1','Test RMSE1','Training RMSE2','Test RMSE2','Training RMSE3','Test RMSE3',...
    'Training RMSE4','Test RMSE4','Training RMSE5','Test RMSE5','Training RMSE6','Test RMSE6','location','northeast');
xlabel('Iteration'); ylabel('RMSE'); axis tight;

