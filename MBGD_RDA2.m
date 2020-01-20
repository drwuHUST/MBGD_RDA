function [RMSEtrain,RMSEtest,mStepSize,stdStepSize]=...
    MBGD_RDA2(XTrain,yTrain,XTest,yTest,alpha,rr,P,numRules,numIt,batchSize)


% alpha: learning rate
% rr: regularization coefficient
% P: dropRule rate
% numRules: number of rules
% numIt: maximum number of iterations

beta1=0.9; beta2=0.999;

[N,M]=size(XTrain); NTest=size(XTest,1);
batchSize=min(N,batchSize);
B=zeros(numRules,M+1); % Rule consequents
% k-means initialization
[ids,C,sumd] = kmeans(XTrain,numRules,'replicate',10);
C=C'; sumd(sumd==0)=mean(sumd); Sigma=repmat(sumd',M,1)/M; 
minSigma=.5*min(Sigma(:));
for r=1:numRules
    B(r,1)=mean(yTrain(ids==r));    
end

%% Iterative update
RMSEtrain=zeros(1,numIt); RMSEtest=RMSEtrain; mStepSize=RMSEtrain; stdStepSize=RMSEtrain;
mC=0; vC=0; mB=0; mSigma=0; vSigma=0; vB=0; yPred=nan(batchSize,1);
for it=1:numIt
    deltaC=zeros(M,numRules); deltaSigma=deltaC;  deltaB=rr*B; deltaB(:,1)=0; % consequent
    f=ones(batchSize,numRules); % firing level of rules
    idsTrain=datasample(1:N,batchSize,'replace',false);
    idsGoodTrain=true(batchSize,1);
    for n=1:batchSize
        idsKeep=rand(1,numRules)<=P;
        f(n,~idsKeep)=0;
        for r=1:numRules
            if idsKeep(r)
                for m=1:M
                    f(n,r)=f(n,r)*exp(-(XTrain(idsTrain(n),m)-C(m,r))^2/(2*Sigma(m,r)^2));
                end
            end
        end
        if ~sum(f(n,:)) % special case: all f(n,:)=0; no dropRule
            idsKeep=~idsKeep;
            f(n,idsKeep)=1;
            for r=1:numRules
                if idsKeep(r)
                    for m=1:M
                        f(n,r)=f(n,r)*exp(-(XTrain(idsTrain(n),m)-C(m,r))^2/(2*Sigma(m,r)^2));
                    end
                end
            end
            idsKeep=true(1,numRules);
        end
        fBar=f(n,:)/sum(f(n,:));
        yR=[1 XTrain(idsTrain(n),:)]*B';
        yPred(n)=fBar*yR'; % prediction
        if isnan(yPred(n))
            %save2base();          return;
            idsGoodTrain(n)=false;
            continue;
        end
        
        % Compute delta
        for r=1:numRules
            if idsKeep(r)
                temp=(yPred(n)-yTrain(idsTrain(n)))*(yR(r)*sum(f(n,:))-f(n,:)*yR')/sum(f(n,:))^2*f(n,r);
                if ~isnan(temp) && abs(temp)<inf
                    % delta of c, sigma, and b
                    for m=1:M
                        deltaC(m,r)=deltaC(m,r)+temp*(XTrain(idsTrain(n),m)-C(m,r))/Sigma(m,r)^2;
                        deltaSigma(m,r)=deltaSigma(m,r)+temp*(XTrain(idsTrain(n),m)-C(m,r))^2/Sigma(m,r)^3;
                        deltaB(r,m+1)=deltaB(r,m+1)+(yPred(n)-yTrain(idsTrain(n)))*fBar(r)*XTrain(idsTrain(n),m);
                    end
                    % delta of b0
                    deltaB(r,1)=deltaB(r,1)+(yPred(n)-yTrain(idsTrain(n)))*fBar(r);
                end
            end
        end
    end
    
    % Training error
    RMSEtrain(it)=sqrt(sum((yTrain(idsTrain(idsGoodTrain))-yPred(idsGoodTrain)).^2)/sum(idsGoodTrain));
    % Test error
    f=ones(NTest,numRules); % firing level of rules
    for n=1:NTest        
        for r=1:numRules % firing levels of rules
            for m=1:M
                f(n,r)=f(n,r)*exp(-(XTest(n,m)-C(m,r))^2/(2*Sigma(m,r)^2));
            end
        end
    end
    yR=[ones(NTest,1) XTest]*B';
    yPredTest=sum(f.*yR,2)./sum(f,2); % prediction
    RMSEtest(it)=sqrt((yTest-yPredTest)'*(yTest-yPredTest)/NTest);
    if isnan(RMSEtest(it)) && it>1
        RMSEtest(it)=RMSEtest(it-1);
    end
    
    % AdaBound
    mC=beta1*mC+(1-beta1)*deltaC;
    vC=beta2*vC+(1-beta2)*deltaC.^2;
    mCHat=mC/(1-beta1^it);
    vCHat=vC/(1-beta2^it);
    mSigma=beta1*mSigma+(1-beta1)*deltaSigma;
    vSigma=beta2*vSigma+(1-beta2)*deltaSigma.^2;
    mSigmaHat=mSigma/(1-beta1^it);
    vSigmaHat=vSigma/(1-beta2^it);
    mB=beta1*mB+(1-beta1)*deltaB;
    vB=beta2*vB+(1-beta2)*deltaB.^2;
    mBHat=mB/(1-beta1^it);
    vBHat=vB/(1-beta2^it);
    % update C, Sigma and B, using AdaBound
    lb=alpha*(1-1/((1-beta2)*it+1));
    ub=alpha*(1+1/((1-beta2)*it));
    lrC=min(ub,max(lb,alpha./(sqrt(vCHat)+10^(-8))));
    C=C-lrC.*mCHat;
    lrSigma=min(ub,max(lb,alpha./(sqrt(vSigmaHat)+10^(-8))));
    Sigma=max(.1*minSigma,Sigma-lrSigma.*mSigmaHat);
    lrB=min(ub,max(lb,alpha./(sqrt(vBHat)+10^(-8))));
    B=B-lrB.*mBHat;
    lr=[lrC(:); lrSigma(:); lrB(:)];
    mStepSize(it)=mean(lr); stdStepSize(it)=std(lr);
end
