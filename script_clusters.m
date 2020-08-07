clc;
close all;
clear all;
data = importdata('E:\sem2\ML\8\USPS_train-2.txt');
input = data(:,1:16);
labels = data(:,17);
iter_accuracy = cell(3,1);
accuracy = cell(3,1);
for s = 1:3
iter_accuracy{s} = zeros(1,50);
end
for s = 1:3
accuracy{s} = zeros(1,16);
end
[r1,c1] = size(input);
%% implementing pca
meaninput = mean(input);
norminput = (input - meaninput);
covinput = cov(norminput);
[eig_vec,eig_val] = eig(covinput);
eigval = (diag(eig_val))';
eigvec = zeros(c1,c1);
for m = 1:c1
    [a,b] = max(eigval);
    eigvec(:,m) = eig_vec(:,b);
    eigval(b) = 0;
end
for k = [5,7,10] %value of k in k-means 
for pc = 1:16 %value of no. of components in PCA  
%% implementing k-means algorithm
newinput = input*eigvec(:,1:pc);
if pc == 16
   newinput = input; 
end 
[r2,c2] = size(newinput);
%%dividing into k random groups  
randomlabels = randi([1 k],r1,1);
rounds = 60;
for mp = 1:rounds
newdata = [newinput randomlabels];
cluster = cell(k,1);
for i = 1:k
aaa = find(newdata(:,pc+1)==i);
cluster{i} = newinput(aaa,:);    
end
%%finding means of randomly divided clusters
meancluster = cell(k,1);
for i = 1:k
meancluster{i} = mean(cluster{i});
end
%%finding distance between means of clusters and input data
distance = zeros(r1,k);
for i = 1:k
for m = 1:r1
distance(m,i) = norm(newinput(m,:) - meancluster{i});
end
end
%%assigning new labels based on distances
for i = 1:r1
[v,ind] = min(distance(i,:));    
randomlabels(i) = ind;  
end
if pc==16
newwdata = [newinput labels randomlabels];
mpmp2 = cell(k,1);
for i = 1:k
mpmp = find(newwdata(:,c2+2)==i);
mpmp2{i} = newwdata(mpmp,1:c2+1);
end     
s=0;
m=0;
for i=1:k
    f=hist(mpmp2{i}(:,c2+1),1:10);
    s=s+sum(f);
    m=m+max(f);
end
if k == 5
iter_accuracy{1}(mp) = m/s;
elseif k == 7
iter_accuracy{2}(mp) = m/s;
elseif k == 10
iter_accuracy{3}(mp) = m/s;
end
end
end
newwdata = [newinput labels randomlabels];
mpmp2 = cell(k,1);
for i = 1:k
mpmp = find(newwdata(:,c2+2)==i);
mpmp2{i} = newwdata(mpmp,1:c2+1);
end
s=0;
m=0;
for i=1:k
    f=hist(mpmp2{i}(:,c2+1),1:10);
    s=s+sum(f);
    m=m+max(f);
end    
if k == 5
accuracy{1}(pc)=m/s;
elseif k == 7
accuracy{2}(pc)=m/s;
elseif k == 10
accuracy{3}(pc)=m/s;
end
end
end
figure
plot(iter_accuracy{1},'r');
hold on;
plot(iter_accuracy{2},'b');
hold on;
plot(iter_accuracy{3},'g');
title('accuracy vs number of iterations');
hold off;
figure
plot(accuracy{1},'r');
hold on;
plot(accuracy{2},'b');
hold on;
plot(accuracy{3},'g');
title('accuracy vs no.of principal components');
hold off;
