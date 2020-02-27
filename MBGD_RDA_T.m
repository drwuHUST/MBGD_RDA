function [RMSEtrain,RMSEtest,A,B,C,D,W,yPredTest]=MBGD_RDA_T(XTrain,yTrain,XTest,yTest,alpha,rr,P,nMFs,nIt,Nbs)

% This function implements a variant of the MBGD-RDA algorithm in the following paper:
%
% Dongrui Wu, Ye Yuan, Jian Huang and Yihua Tan, "Optimize TSK Fuzzy Systems for Regression Problems: 
% Mini-Batch Gradient Descent with Regularization, DropRule and AdaBound (MBGD-RDA)," IEEE Trans. 
% on Fuzzy Systems, 2020, accepted.
%
% It specifies the number of trapezoidal MFs in each input domain by nMFs.
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
% A,B,C,D: M*nMFs matrices specifying the a, b, c, d parameters of the trapezoidal MFs. See derivations.pdf
% W: nRules*(M+1) matrix of the consequent parameters for the rules. nRules=nMFs^M.

beta1=0.9; beta2=0.999;

[N,M]=size(XTrain); NTest=size(XTest,1);
if Nbs>N; Nbs=N; end
nMFsVec=nMFs*ones(M,1);
nRules=nMFs^M; % number of rules
points=zeros(M,nMFs+3); W=zeros(nRules,M+1);
for m=1:M % Initialization
    points(m,:)=linspace(min(XTrain(:,m)),max(XTrain(:,m)),nMFs+3);
end
A=points(:,1:end-3); B=points(:,2:end-2); C=points(:,3:end-1); D=points(:,4:end);

%% Iterative update
mu=zeros(M,nMFs);  RMSEtrain=zeros(1,nIt); RMSEtest=RMSEtrain;
mA=0; vA=0; mB=0; vB=0; mC=0; vC=0; mD=0; vD=0; mW=0; vW=0; yPred=nan(Nbs,1);
for it=1:nIt
    deltaA=zeros(M,nMFs); deltaB=deltaA;  deltaC=deltaA; deltaD=deltaA;  deltaW=rr*W; deltaW(:,1)=0; % consequent
    f=ones(Nbs,nRules); % firing level of rules
    idsTrain=datasample(1:N,Nbs,'replace',false);
    idsGoodTrain=true(Nbs,1);
    for n=1:Nbs
        for m=1:M % membership grades of MFs
            mu(m,:)=MG(XTrain(idsTrain(n),m)*ones(1,nMFs),[A(m,:); B(m,:); C(m,:); D(m,:)]);
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
                        if XTrain(idsTrain(n),m)>A(m,vec(m)) && XTrain(idsTrain(n),m)<B(m,vec(m))
                            deltaA(m,vec(m))=deltaA(m,vec(m))+temp*(XTrain(idsTrain(n),m)-B(m,vec(m)))/...
                                (MG(XTrain(idsTrain(n),m),[A(m,vec(m)); B(m,vec(m)); C(m,vec(m)); D(m,vec(m))])...
                                *(B(m,vec(m))-A(m,vec(m)))^2);
                            deltaB(m,vec(m))=deltaB(m,vec(m))-temp/(B(m,vec(m))-A(m,vec(m)));
                        end
                        if XTrain(idsTrain(n),m)>C(m,vec(m)) && XTrain(idsTrain(n),m)<D(m,vec(m))
                            deltaD(m,vec(m))=deltaD(m,vec(m))+temp*(XTrain(idsTrain(n),m)-C(m,vec(m)))/...
                                (MG(XTrain(idsTrain(n),m),[A(m,vec(m)); B(m,vec(m)); C(m,vec(m)); D(m,vec(m))])...
                                *(D(m,vec(m))-C(m,vec(m)))^2);
                            deltaC(m,vec(m))=deltaC(m,vec(m))+temp/(D(m,vec(m))-C(m,vec(m)));
                        end
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
    
    mA=beta1*mA+(1-beta1)*deltaA;
    vA=beta2*vA+(1-beta2)*deltaA.^2;
    mAHat=mA/(1-beta1^it);
    vAHat=vA/(1-beta2^it);
    lrA=min(ub,max(lb,alpha./(sqrt(vAHat)+10^(-8))));
    A=A-lrA.*mAHat;
    
    mB=beta1*mB+(1-beta1)*deltaB;
    vB=beta2*vB+(1-beta2)*deltaB.^2;
    mBHat=mB/(1-beta1^it);
    vBHat=vB/(1-beta2^it);
    lrB=min(ub,max(lb,alpha./(sqrt(vBHat)+10^(-8))));
    B=B-lrB.*mBHat;
    
    mC=beta1*mC+(1-beta1)*deltaC;
    vC=beta2*vC+(1-beta2)*deltaC.^2;
    mCHat=mC/(1-beta1^it);
    vCHat=vC/(1-beta2^it);
    lrC=min(ub,max(lb,alpha./(sqrt(vCHat)+10^(-8))));
    C=C-lrC.*mCHat;
    
    mD=beta1*mD+(1-beta1)*deltaD;
    vD=beta2*vD+(1-beta2)*deltaD.^2;
    mDHat=mD/(1-beta1^it);
    vDHat=vD/(1-beta2^it);
    lrD=min(ub,max(lb,alpha./(sqrt(vDHat)+10^(-8))));
    D=D-lrD.*mDHat;
    
    mW=beta1*mW+(1-beta1)*deltaW;
    vW=beta2*vW+(1-beta2)*deltaW.^2;
    mWHat=mW/(1-beta1^it);
    vWHat=vW/(1-beta2^it);
    lrW=min(ub,max(lb,alpha./(sqrt(vWHat)+10^(-8))));
    W=W-lrW.*mWHat;
    
    % Adjust the rank to make sure a<=b<=c<=d
    for m=1:M
        for r=1:nMFs
            abcd=sort([A(m,r) B(m,r) C(m,r) D(m,r)]);
            A(m,r)=abcd(1); B(m,r)=abcd(2);
            C(m,r)=abcd(3); D(m,r)=abcd(4);
        end
    end
    
    % Training RMSE
    RMSEtrain(it)=sqrt(sum((yTrain(idsTrain(idsGoodTrain))-yPred(idsGoodTrain)).^2)/sum(idsGoodTrain));
    % Test RMSE
    f=ones(NTest,nRules); % firing level of rules
    for n=1:NTest
        for m=1:M % membership grades of MFs
            mu(m,:)=MG(XTest(n,m)*ones(1,nMFs),[A(m,:); B(m,:); C(m,:); D(m,:)]);
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
    yPredTest(isnan(yPredTest))=nanmean(yPredTest);
    RMSEtest(it)=sqrt((yTest-yPredTest)'*(yTest-yPredTest)/NTest);
    if isnan(RMSEtest(it)) && it>1
        RMSEtest(it)=RMSEtest(it-1);
    end
end
end

function mu=MG(x,abcd)
% if abcd(1)==abcd(2); abcd(2)=abcd(1)+2*eps; end
% if abcd(4)==abcd(3); abcd(4)=abcd(3)+2*eps; end
mu=max(0,min(1,min((x-abcd(1,:))./(abcd(2,:)-abcd(1,:)),(abcd(4,:)-x)./(abcd(4,:)-abcd(3,:)))));
end

function vec=idx2vec(idx,nMFs)
% Convert from a scalar index of the rule to a vector indices of MFs
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