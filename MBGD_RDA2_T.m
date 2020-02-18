function [RMSEtrain,RMSEtest,A,B,C,D,W]=MBGD_RDA2_T(XTrain,yTrain,XTest,yTest,alpha,rr,P,nRules,nIt,Nbs)

% Use trapezoidal MFs

% alpha: learning rate
% rr: regularization coefficient
% P: dropRule rate
% nRules: number of rules
% nIt: maximum number of iterations
% Nbs: batch size

beta1=0.9; beta2=0.999;

[N,M]=size(XTrain); NTest=size(XTest,1);
Nbs=min(N,Nbs);
W=zeros(nRules,M+1); % Rule consequents
% k-means initialization
[ids,C] = kmeans(XTrain,nRules,'replicate',3);
Sigma=C;
for r=1:nRules
    Sigma(r,:)=std(XTrain(ids==r,:));
    W(r,1)=mean(yTrain(ids==r));
end
Sigma(Sigma==0)=mean(Sigma(:));
A=C-Sigma; D=C+Sigma; B=C-.25*Sigma; C=C+.25*Sigma;

%% Iterative update
RMSEtrain=zeros(1,nIt); RMSEtest=RMSEtrain; 
mA=0; vA=0; mB=0; vB=0; mC=0; vC=0; mD=0; vD=0;
mW=0; vW=0; yPred=nan(Nbs,1);
for it=1:nIt
    deltaA=zeros(nRules,M); deltaB=deltaA;  deltaC=deltaA; deltaD=deltaA; deltaW=rr*W; deltaW(:,1)=0; % consequent
    f=ones(Nbs,nRules); % firing level of rules
    idsTrain=datasample(1:N,Nbs,'replace',false);
    idsGoodTrain=true(Nbs,1);
    for n=1:Nbs
        idsKeep=rand(1,nRules)<=P;
        f(n,~idsKeep)=0;
        for r=1:nRules
            if idsKeep(r)
                f(n,r)=prod(MG(XTrain(idsTrain(n),:),[A(r,:); B(r,:); C(r,:); D(r,:)]));
            end
        end
        if ~sum(f(n,:)) % special case: all f(n,:)=0; no dropRule
            idsKeep=~idsKeep;
            f(n,idsKeep)=1;
            for r=1:nRules
                if idsKeep(r)
                    f(n,r)=prod(MG(XTrain(idsTrain(n),:),[A(r,:); B(r,:); C(r,:); D(r,:)]));
                end
            end
            idsKeep=true(1,nRules);
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
                    % delta of c, sigma, and b
                    for m=1:M
                        if XTrain(idsTrain(n),m)>A(r,m) && XTrain(idsTrain(n),m)<B(r,m)
                            deltaA(r,m)=deltaA(r,m)+temp*(XTrain(idsTrain(n),m)-B(r,m))/...
                                (MG(XTrain(idsTrain(n),m),[A(r,m); B(r,m); C(r,m); D(r,m)])*(B(r,m)-A(r,m))^2);
                            deltaB(r,m)=deltaB(r,m)-temp/(B(r,m)-A(r,m));
                        end
                        if XTrain(idsTrain(n),m)>C(r,m) && XTrain(idsTrain(n),m)<D(r,m)
                            deltaD(r,m)=deltaD(r,m)+temp*(XTrain(idsTrain(n),m)-C(r,m))/...
                                (MG(XTrain(idsTrain(n),m),[A(r,m); B(r,m); C(r,m); D(r,m)])*(D(r,m)-C(r,m))^2);
                            deltaC(r,m)=deltaC(r,m)+temp/(D(r,m)-C(r,m));
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
    for r=1:nRules
        for m=1:M
            abcd=sort([A(r,m) B(r,m) C(r,m) D(r,m)]);
            A(r,m)=abcd(1); B(r,m)=abcd(2);
            C(r,m)=abcd(3); D(r,m)=abcd(4);
        end
    end
    
    % Training RMSE
    RMSEtrain(it)=sqrt(sum((yTrain(idsTrain(idsGoodTrain))-yPred(idsGoodTrain)).^2)/sum(idsGoodTrain));
    % Test RMSE
    f=ones(NTest,nRules); % firing level of rules
    for n=1:NTest
        for r=1:nRules
            f(n,r)= prod(MG(XTest(n,:),[A(r,:); B(r,:); C(r,:); D(r,:)]));
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
