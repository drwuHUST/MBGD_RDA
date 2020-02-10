clc; clearvars; close all; %rng(0);

Mm=2; % number of MFs in each input domain
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
% Specify the total number of rules; use the original features without dimensionality reduction
nRules=30; % number of rules
[RMSEtrain1,RMSEtest1]=MBGD_RDA2(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nRules,nIt,Nbs);

% Dimensionality reduction
maxFeatures=5; % maximum number of features to use
if M>maxFeatures
    [~,XPCA,latent]=pca(X);
    realDim98=find(cumsum(latent)>=.98*sum(latent),1,'first');
    usedDim=min(maxFeatures,realDim98);
    X=XPCA(:,1:usedDim); [N0,M]=size(X);
end
nRules=Mm^M; % number of rules
XTrain=X(idsTrain,:); XTest=X; XTest(idsTrain,:)=[]; 

% Specify the number of MFs in each input domain; Assume x1 has two MFs
% X1_1 and X1_2; then, all rules involving the first FS of x1 use the same
% X1_1, and all rules involving the second FS of x1 use the same X1_2
[RMSEtrain2,RMSEtest2]=MBGD_RDA(XTrain,yTrain,XTest,yTest,alpha,lambda,P,Mm,nIt,Nbs);

% Specify the total number of rules; each rule uses different membership functions
[RMSEtrain3,RMSEtest3]=MBGD_RDA2(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nRules,nIt,Nbs);


%% Plot results
figure('Position', get(0, 'Screensize')); 
plot(RMSEtrain1,'r:','linewidth',2); hold on;
plot(RMSEtest1,'r-','linewidth',2);
plot(RMSEtrain2,'k:','linewidth',2);
plot(RMSEtest2,'k-','linewidth',2);
plot(RMSEtrain3,'b:','linewidth',2);
plot(RMSEtest3,'b-','linewidth',2);
legend('Training RMSE1','Test RMSE1','Training RMSE2','Test RMSE2','Training RMSE3','Test RMSE3','location','northeast');
xlabel('Iteration'); ylabel('RMSE');

