function [RMSEtrain,RMSEtest,C,Sigma,W]=MBGD_RDA(XTrain,yTrain,XTest,yTest,alpha,rr,P,nMFs,nIt,Nbs)

% This function implements the MBGD-RDA algorithm in the following paper:
%
% Dongrui Wu, Ye Yuan, Jian Huang and Yihua Tan, "Optimize TSK Fuzzy Systems for Regression Problems: 
% Mini-Batch Gradient Descent with Regularization, DropRule and AdaBound (MBGD-RDA)," IEEE Trans. 
% on Fuzzy Systems, 2020, accepted.
%
% It specifies the number of Gaussian MFs in each input domain by nMFs.
% Assume x1 has two MFs X1_1 and X1_2; then, all rules involving the first FS of x1 use the same X1_1,
% and all rules involving the second FS of x1 use the same X1_2
%
% By Dongrui Wu, drwu@hust.edu.cn
%
% %% Inputs:
% XTrain: N*M matrix of the training inputs. N is the number of samples, and M the feature dimensionality.
% yTrain: N*1 vector of the labels for XTrain
% XTest: NTest*M matrix of the test inputs
% yTest: NTest*1 vector of the labels for XTest
% alpha: scalar, learning rate
% rr: scalar, L2 regularization coefficient 
% P: scalar in [0.5, 1), dropRule rate
% nMFs: scalar in [2, 5], number of MFs in each input domain
% nIt: scalar, maximum number of iterations
% Nbs: batch size. typically 32 or 64
%
% %% Outputs:
% RMSEtrain: 1*nIt vector of the training RMSE at different iterations
% RMSEtest: 1*nIt vector of the test RMSE at different iterations
% C: M*nMFs matrix of the centers of the Gaussian MFs
% Sigma: M*nMFs matrix of the standard deviations of the Gaussian MFs
% W: nRules*(M+1) matrix of the consequent parameters for the rules. nRules=nMFs^M.

beta1=0.9; beta2=0.999;

[N,M]=size(XTrain); NTest=size(XTest,1);
if Nbs>N; Nbs=N; end
nMFsVec=nMFs*ones(M,1);
nRules=nMFs^M; % number of rules
C=zeros(M,nMFs); Sigma=C; W=zeros(nRules,M+1);
for m=1:M % Initialization
    C(m,:)=linspace(min(XTrain(:,m)),max(XTrain(:,m)),nMFs);
    Sigma(m,:)=10*std(XTrain(:,m));
end
minSigma=0.01*min(Sigma(:));
maxSigma=10*max(Sigma(:));

%% Iterative update
mu=zeros(M,nMFs);  RMSEtrain=zeros(1,nIt); RMSEtest=RMSEtrain;
mC=0; vC=0; mW=0; mSigma=0; vSigma=0; vW=0; yPred=nan(Nbs,1);
for it=1:nIt
    deltaC=zeros(M,nMFs); deltaSigma=deltaC;  deltaW=rr*W; deltaW(:,1)=0; % consequent
    f=ones(Nbs,nRules); % firing level of rules
    idsTrain=datasample(1:N,Nbs,'replace',false);
    idsGoodTrain=true(Nbs,1);
    for n=1:Nbs
        for m=1:M % membership grades of MFs
            mu(m,:)=exp(-(XTrain(idsTrain(n),m)-C(m,:)).^2./(2*Sigma(m,:).^2));
        end
        
        idsKeep=rand(1,nRules)<=P;
        f(n,~idsKeep)=0;
        for r=1:nRules
            if idsKeep(r)
                idsMFs=idx2vec(r,nMFsVec);
                for m=1:M
                    f(n,r)=f(n,r)*mu(m,idsMFs(m));
                end
            end
        end
        if ~sum(f(n,:)) % special case: all f(n,:)=0; no dropRule
            idsKeep=true(1,nRules);
            f(n,:)=1;
            for r=1:nRules
                idsMFs=idx2vec(r,nMFsVec);
                for m=1:M
                    f(n,r)=f(n,r)*mu(m,idsMFs(m));
                end
            end
        end
        fBar=f(n,:)/sum(f(n,:));
        yR=[1 XTrain(idsTrain(n),:)]*W';
        yPred(n)=fBar*yR'; % prediction
        if isnan(yPred(n))
            %save2base();          return;
            idsGoodTrain(n)=false;
            continue;
        end
        
        % Compute delta
        for r=1:nRules
            if idsKeep(r)
                temp=(yPred(n)-yTrain(idsTrain(n)))*(yR(r)*sum(f(n,:))-f(n,:)*yR')/sum(f(n,:))^2*f(n,r);
                if ~isnan(temp) && abs(temp)<inf
                    vec=idx2vec(r,nMFsVec);
                    % delta of c, sigma, and b
                    for m=1:M
                        deltaC(m,vec(m))=deltaC(m,vec(m))+temp*(XTrain(idsTrain(n),m)-C(m,vec(m)))/Sigma(m,vec(m))^2;
                        deltaSigma(m,vec(m))=deltaSigma(m,vec(m))+temp*(XTrain(idsTrain(n),m)-C(m,vec(m)))^2/Sigma(m,vec(m))^3;
                        deltaW(r,m+1)=deltaW(r,m+1)+(yPred(n)-yTrain(idsTrain(n)))*fBar(r)*XTrain(idsTrain(n),m);
                    end
                    % delta of b0
                    deltaW(r,1)=deltaW(r,1)+(yPred(n)-yTrain(idsTrain(n)))*fBar(r);
                end
            end
        end
    end
    
    % AdaBound
    lb=alpha*(1-1/((1-beta2)*it+1));
    ub=alpha*(1+1/((1-beta2)*it));
    
    mC=beta1*mC+(1-beta1)*deltaC;
    vC=beta2*vC+(1-beta2)*deltaC.^2;
    mCHat=mC/(1-beta1^it);
    vCHat=vC/(1-beta2^it);
    lrC=min(ub,max(lb,alpha./(sqrt(vCHat)+10^(-8))));
    C=C-lrC.*mCHat;
    
    mSigma=beta1*mSigma+(1-beta1)*deltaSigma;
    vSigma=beta2*vSigma+(1-beta2)*deltaSigma.^2;
    mSigmaHat=mSigma/(1-beta1^it);
    vSigmaHat=vSigma/(1-beta2^it);
    lrSigma=min(ub,max(lb,alpha./(sqrt(vSigmaHat)+10^(-8))));
    Sigma=min(maxSigma,max(minSigma,Sigma-lrSigma.*mSigmaHat));
    
    mW=beta1*mW+(1-beta1)*deltaW;
    vW=beta2*vW+(1-beta2)*deltaW.^2;
    mWHat=mW/(1-beta1^it);
    vWHat=vW/(1-beta2^it);
    lrW=min(ub,max(lb,alpha./(sqrt(vWHat)+10^(-8))));
    W=W-lrW.*mWHat;
    
    % Training RMSE
    RMSEtrain(it)=sqrt(sum((yTrain(idsTrain(idsGoodTrain))-yPred(idsGoodTrain)).^2)/sum(idsGoodTrain));
    % Test RMSE
    f=ones(NTest,nRules); % firing level of rules
    for n=1:NTest
        for m=1:M % membership grades of MFs
            mu(m,:)=exp(-(XTest(n,m)-C(m,:)).^2./(2*Sigma(m,:).^2));
        end
        
        for r=1:nRules % firing levels of rules
            idsMFs=idx2vec(r,nMFsVec);
            for m=1:M
                f(n,r)=f(n,r)*mu(m,idsMFs(m));
            end
        end
    end
    yR=[ones(NTest,1) XTest]*W';
    yPredTest=sum(f.*yR,2)./sum(f,2); % prediction
    RMSEtest(it)=sqrt((yTest-yPredTest)'*(yTest-yPredTest)/NTest);
    if isnan(RMSEtest(it)) && it>1
        RMSEtest(it)=RMSEtest(it-1);
    end
end
end

function vec=idx2vec(idx,nMFs)
% Convert from a scalar index of the rule to a vector index of MFs
vec=zeros(1,length(nMFs));
prods=[1; cumprod(nMFs(end:-1:1))];
if idx>prods(end)
    error('Error: idx is larger than the number of rules.');
end
prev=0;
for i=1:length(nMFs)
    vec(i)=floor((idx-1-prev)/prods(end-i))+1;
    prev=prev+(vec(i)-1)*prods(end-i);
end
end
