clc
clear all

% ---------------------------
issave_training = true;
issave_examples = true;

%---------------------------
load('data.mat')
wname='db1';
[x_train_A,x_train_D] = dwt(x_train(:,1),wname);
[x_vali_A,x_vali_D] = dwt(x_vali(:,1),wname);
[x_test_A,x_test_D] = dwt(x_test(:,1),wname);
% train
for ii=1:size(f_train,1)
sig=f_train(ii,:);
[cA,cD] = dwt(sig,wname);
f_train_A(ii,:)=cA;
f_train_D(ii,:)=cD;
end
for ii=1:size(u_train,1)
sig=u_train(ii,:);
[cA,cD] = dwt(sig,wname);
u_train_A(ii,:)=cA;
u_train_D(ii,:)=cD;
end
% vali
for ii=1:size(f_vali,1)
sig=f_vali(ii,:);
[cA,cD] = dwt(sig,wname);
f_vali_A(ii,:)=cA;
f_vali_D(ii,:)=cD;
end
for ii=1:size(u_vali,1)
sig=u_vali(ii,:);
[cA,cD] = dwt(sig,wname);
u_vali_A(ii,:)=cA;
u_vali_D(ii,:)=cD;
end
% test
for ii=1:size(f_test,1)
sig=f_test(ii,:);
[cA,cD] = dwt(sig,wname);
f_test_A(ii,:)=cA;
f_test_D(ii,:)=cD;
end
for ii=1:size(u_test,1)
sig=u_test(ii,:);
[cA,cD] = dwt(sig,wname);
u_test_A(ii,:)=cA;
u_test_D(ii,:)=cD;
end
if issave_training
    saveFolder = ['../Data/'];
    saveName = ['data.mat'];
    if ~isdir(saveFolder)
        mkdir(saveFolder);
    end
    save([saveFolder, saveName],...
        'f_train_A','f_train_D','x_train_A','x_train_D','u_train_A','u_train_D', ...
        'f_vali_A','f_vali_D','x_vali_A','x_vali_D','u_vali_A','u_vali_D',...
        'f_test_A','f_test_D','x_test_A','x_test_D','u_test_A','u_test_D');
end








