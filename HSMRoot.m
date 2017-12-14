function [data_output] = HSMRoot(t, learning_param, data, market, noise, h, rob)

%t: periods to iterate
%learning_param: beta, eta, delta (learning parameters)
%data: vector of input data
%noise: optional vector of noise in the model
%h: vector containing the set of available heuristics
%rob: whether or not robot fundamentalists are present in the market
%e.g.: HSMRoot(50, [0.4, 0.7, 0.9], infl, [0.05, 3], noise, [1 1 1 1 0 0], 0)

%Parameter input
beta = learning_param(1);
eta = learning_param(2);
delta = learning_param(3);

rf = market(1);
y = market(2);
lower = market(3);
upper = market(4);

pf = y/rf;

%Time and heuristic count
T = size(data, 1);
H = length(h);

%Heuristic parameters
w = 0.65;
gwtr = 0.4;
gstr = 1.3;

%Initialisation
robots = zeros(t, 1);

pexp = zeros(t, H);
u = zeros(t, H);
unorm = zeros(t, H);
n = zeros(t, H);

new = zeros(t, H);

data_output.price = zeros(t, 1);
data_output.robots = zeros(t, 1);
data_output.predictions = zeros(t, 1);
data_output.fraction = zeros(t, H);

%First values
pexp(3, 1) = data(1, 1); %Adaptive Heuristic
n(3, :) = h(1, :)/sum(h);

%Learning model
for i = 3:t
    if rob
       %Determine the number of robot fundamentalists
       robots(i, 1) = 1 - exp(-abs(data(i-1, 1) - pf)/200);
    end
    
    %Define heuristic predictions based on data
    pexp(i+1, 1) = w * data(i-1, 1) + (1 - w) * pexp(i, 1); %Adaptive Heuristic (ADA)
    pexp(i+1, 2) = data(i-1, 1) + gwtr * (data(i-1, 1) - data(i-2, 1)); % Weak Trend-following Heuristic (WTR)
    pexp(i+1, 3) = data(i-1, 1) + gstr * (data(i-1, 1) - data(i-2, 1)); % Strong Trend-following Heuristic (STR)
    
    ave = mean(data(1:i-1, 1));
    
    pexp(i+1, 4) = 0.5 * ave + 1.5 * data(i-1, 1) - data(i-2, 1); %Anchoring and Adjustment Rule with Learning Anchor (LAA)
    pexp(i+1, 5) = data(i-1, 1); %Naive expectations
    pexp(i+1, 6) = pf; %Fundamental expectations
    
    for j = 1:H
        pexp(i+1, j) = max(lower, min(upper, pexp(i+1, j)));
    end
    
    %Find new price
    avepred = n(i, :) * pexp(i+1, :)';
    data_output.price(i, 1) = ((1 - robots(i, 1)) * avepred + robots(i, 1) * pf + y + noise(i, 1))/(1 + rf);
    
    %Out-of-sample periods
    if (i>T)
       data(i, 1) = data_output.price(i, 1); 
    end
    
    %Update performance of heuristics
    for j = 1:H
       if (h(1, j) == 0)
           n(i+1, j) = 0;
       else
           if (i>3)
               u(i, j) = eta * u(i-1, j) - (data(i, 1) - pexp(i, j)).^2;
           end
       end
    end
    
    umax = max(-u(i, :));
    
    for j = 1:H
        if (h(1, j) == 1)
            unorm(i, j) = u(i, j) + umax;
        end
    end
    
    EPS = eps();
    %eps = 0.00000000000000000000000000000000000000000000000000000000001;
    nsum = 0;
   
    for j = 1:H
        if (h(1, j) == 1)
            z = 0;
            for k = 1:H
                z = z + exp(beta * (unorm(i, k) - unorm(i, j)));
            end
            new(i, j) = min(1 - EPS, max(EPS, 1/z));
            nsum = nsum + new(i, j);
        end
    end
    
    for j = 1:H
        if (h(1, j) == 1)
            n(i+1, j) = delta * n(i, j) + (1 - delta) * new(i, j)/nsum;
        end
    end
end

data_output.robots = robots;
data_output.predictions = pexp;
data_output.fraction = n;
data_output.prederror(:, 1) = (data_output.price(:, 1) - data(:, 1));
mse = mean(data_output.prederror(5:T, 1).^2);