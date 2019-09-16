clc; clearvars; close all; rng(0);

Mm=2; % number of MFs in each input domain
alpha=.01; % initial learning rate
Nbs=64; % batch size
lambda=0.05; % L2 regularization coefficient
P=0.5; % DropRule rate
nIt=200; % number of iterations
maxFeatures=5; % maximum number of features to use

temp=load('PM10.mat'); data=temp.data;
X0=data(:,1:end-1); y0=data(:,end); y0=y0-mean(y0);
X0 = zscore(X0); [N0,M]=size(X0);
if M>maxFeatures
    [~,XPCA,latent]=pca(X0);
    realDim98=find(cumsum(latent)>=.98*sum(latent),1,'first');
    usedDim=min(maxFeatures,realDim98);
    X0=XPCA(:,1:usedDim); [N0,M]=size(X0);
end
N=round(N0*.7);

idsTrain=datasample(1:N0,N,'replace',false);
XTrain=X0(idsTrain,:); yTrain=y0(idsTrain);
XTest=X0; XTest(idsTrain,:)=[]; yTest=y0; yTest(idsTrain)=[];
[RMSEtrain,RMSEtest]=MBGD_RDA(XTrain,yTrain,XTest,yTest,alpha,lambda,P,Mm,nIt,Nbs);

figure; 
plot(RMSEtrain,'k:','linewidth',2); hold on;
plot(RMSEtest,'g-','linewidth',2);
legend('Training RMSE','Test RMSE','location','northeast');
xlabel('Iteration'); ylabel('RMSE');

