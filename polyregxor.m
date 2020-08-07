clc;
close all;
clear all;

%% inputs and outputs of xor gate 
x = [0,0;
    0,1;
    1,0;
    1,1];
y = [0;1;1;0];

%% adding column of ones to X so that it will be in form 1+x, 
%so when we multiply that with weights it will be in form y =w0 + wx =  w0 + w1*x1 + w2*x2
%so w0 is bias and w1 and w2 are weights if it is linear regression

in = [ones(length(x),1) x];

%% polynomial regression
in = [in, x.^2];
[xs,ys] = size(in);

%taking initial weights 
w = rand(ys,1);
ye = in*w;  %Expected Output

%% weights updating
for n = 1:2000
    w  = w - 0.02*(in'*(ye-y));
    ye = in*w;   
end
ye


       
        
        
