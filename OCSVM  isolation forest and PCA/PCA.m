clear;clc;close all
load data_all.mat
%%
X_mean = mean(x_train);  %������x_trainƽ��ֵ
X_std = std(x_train);    %���׼��
[X_row,X_col] = size(x_train); %��Xtrain�С�����
x_train=(x_train-X_mean)./X_std;
%��Э�������
sigmax_train = cov(x_train);
%��Э���������������ֽ⣬lamdaΪ����ֵ���ɵĶԽ���T����Ϊ��λ��������������lamda�е�����ֵһһ��Ӧ��
[T,lamda] = eig(sigmax_train);

%ȡ�Խ�Ԫ��(���Ϊһ������)����lamdaֵ�������·�תʹ��Ӵ�С���У���Ԫ������ֵΪ1�����ۼƹ�����С��90%��������Ԫ����
D = flipud(diag(lamda));
num_pc = 1;
while sum(D(1:num_pc))/sum(D) < 0.9
    num_pc = num_pc +1;
end

%ȡ��lamda���Ӧ����������
P = T(:,X_col-num_pc+1:X_col);
TT=x_train*T;
TT1=x_train*P;
%���Ŷ�Ϊ99%��Qͳ�ƿ�����
alpha=0.01;
for i = 1:3
    theta(i) = sum((D(num_pc+1:X_col)).^i);
end
h0 = 1 - 2*theta(1)*theta(3)/(3*theta(2)^2);
ca = norminv(1-alpha,0,1);
QUCL = theta(1)*(h0*ca*sqrt(2*theta(2))/theta(1) + 1 + theta(2)*h0*(h0 - 1)/theta(1)^2)^(1/h0);                               

%�����Ŷ�Ϊ99%��95%ʱ��T2ͳ�ƿ�����                       
T2UCL1=num_pc*(X_row-1)*(X_row+1)*finv(1-alpha,num_pc,X_row - num_pc)/(X_row*(X_row - num_pc));
%% train data
%��׼������
n = size(x_train,1);

%��T2ͳ������Qͳ����
[r,y] = size(P*P');
I = eye(r,y);

T2train = zeros(n,1);
Qtrain = zeros(n,1);
for i = 1:n
    T2train(i)=x_train(i,:)*P*pinv(lamda(X_col-num_pc+1:X_col,X_col-num_pc+1:X_col))*P'*x_train(i,:)';  
    Qtrain(i) = x_train(i,:)*(I - P*P')*(I - P*P')'*x_train(i,:)';                                                                                    
end
%% valdiation data
%��׼������
n = size(x_val,1);
x_val=(x_val-repmat(X_mean,n,1))./repmat(X_std,n,1);

%��T2ͳ������Qͳ����
[r,y] = size(P*P');
I = eye(r,y);

T2val = zeros(n,1);
Qval = zeros(n,1);
for i = 1:n
    T2val(i)=x_val(i,:)*P*pinv(lamda(X_col-num_pc+1:X_col,X_col-num_pc+1:X_col))*P'*x_val(i,:)';  
    Qval(i) = x_val(i,:)*(I - P*P')*(I - P*P')'*x_val(i,:)';                                                                                    
end
%% test data
%��׼������
n = size(x_test,1);
x_test=(x_test-repmat(X_mean,n,1))./repmat(X_std,n,1);

%��T2ͳ������Qͳ����
[r,y] = size(P*P');
I = eye(r,y);

T2test = zeros(n,1);
Qtest = zeros(n,1);
for i = 1:n
    T2test(i)=x_test(i,:)*P*pinv(lamda(X_col-num_pc+1:X_col,X_col-num_pc+1:X_col))*P'*x_test(i,:)';  
    Qtest(i) = x_test(i,:)*(I - P*P')*(I - P*P')'*x_test(i,:)';                                                                                    
end
%% abnormal data
%��׼������
n = size(x_outlier,1);
x_outlier=(x_outlier-repmat(X_mean,n,1))./repmat(X_std,n,1);

%��T2ͳ������Qͳ����
[r,y] = size(P*P');
I = eye(r,y);

T2ab= zeros(n,1);
Qoutlier = zeros(n,1);
for i = 1:n
    T2ab(i)=x_outlier(i,:)*P*pinv(lamda(X_col-num_pc+1:X_col,X_col-num_pc+1:X_col))*P'*x_outlier(i,:)';  
    Qoutlier(i) = x_outlier(i,:)*(I - P*P')*(I - P*P')'*x_outlier(i,:)';                                                                                    
end

%% accuracy
acctrain_spe=sum(Qtrain<QUCL)/length(Qtrain);
accval_spe=sum(Qval<QUCL)/length(Qval);
acctest_spe=sum(Qtest<QUCL)/length(Qtest);
accoutlier_spe=sum(Qoutlier>QUCL)/length(Qoutlier);

acctrain_t2=sum(T2train<T2UCL1)/length(Qtrain);
accval_t2=sum(T2val<T2UCL1)/length(Qval);
acctest_t2=sum(T2test<T2UCL1)/length(Qtest);
accoutlier_t2=sum(T2ab>T2UCL1)/length(Qoutlier);
%% SPE
figure
subplot(2,2,1)
plot(1:size(x_train,1),Qtrain,'k');
xlabel('������');
ylabel('SPE');
title('training')
hold on;
line([0,size(x_train,1)],[QUCL,QUCL],'LineStyle','--','Color','r');
subplot(2,2,2)
plot(1:size(x_val,1),Qval,'k');
xlabel('������');
ylabel('SPE');
hold on;
line([0,size(x_val,1)],[QUCL,QUCL],'LineStyle','--','Color','r');
title('validation')
subplot(2,2,3)
plot(1:size(x_test,1),Qtest,'k');
xlabel('������');
ylabel('SPE');
hold on;
line([0,size(x_test,1)],[QUCL,QUCL],'LineStyle','--','Color','r');
title('test')
subplot(2,2,4)
plot(1:size(x_outlier,1),Qoutlier,'k');
xlabel('������');
ylabel('SPE');
hold on;
line([0,size(x_outlier,1)],[QUCL,QUCL],'LineStyle','--','Color','r');
title('abnormal')
%% T2
figure
subplot(2,2,1)
plot(1:size(x_train,1),T2train,'k');
xlabel('������');
ylabel('T2');
title('training')
hold on;
line([0,size(x_train,1)],[T2UCL1,T2UCL1],'LineStyle','--','Color','r');
subplot(2,2,2)
plot(1:size(x_val,1),T2val,'k');
xlabel('������');
ylabel('T2');
hold on;
line([0,size(x_val,1)],[T2UCL1,T2UCL1],'LineStyle','--','Color','r');
title('validation')
subplot(2,2,3)
plot(1:size(x_test,1),T2test,'k');
xlabel('������');
ylabel('T2');
hold on;
line([0,size(x_test,1)],[T2UCL1,T2UCL1],'LineStyle','--','Color','r');
title('test')
subplot(2,2,4)
plot(1:size(x_outlier,1),T2ab,'k');
xlabel('������');
ylabel('T2');
hold on;
line([0,size(x_outlier,1)],[T2UCL1,T2UCL1],'LineStyle','--','Color','r');
title('abnormal')
%% 
save PCA_156.mat
