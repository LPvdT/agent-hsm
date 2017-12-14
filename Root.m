%% Load data vectors
ImportData();
data_size = size(infl_NL, 1);
T = data_size;
noise = transpose(rednoise(data_size));
infl_unstable = transpose(rednoise(data_size));

%% Variable declarations
rf = 1;
lower = -1000;
upper = 1000;

robots = zeros(data_size, 1);

pexp = zeros(data_size, 1);
price = zeros(data_size, 1);

mse = zeros(data_size, 14);

error = zeros(data_size, 1);

x = zeros(data_size, 1);
xout = zeros(data_size, 1);

outerr = zeros(data_size, 1);
outerrbench = zeros(data_size,1);
arpred = zeros(data_size,1);
arerror = zeros(data_size,1);

ar2beta = zeros(data_size,1);

xoutmov = zeros(data_size,1);

msein = zeros(data_size, 1);
mseinbench = zeros(data_size, 1);
mseinar = zeros(data_size, 1);
outerrmov1 = zeros(data_size, 1);
outerrmov2 = zeros(data_size, 1);
outerrmov = zeros(data_size, 1);
outerrmovbench1 = zeros(data_size, 1);
outerrmovbench2 = zeros(data_size, 1);
outerrmovbench = zeros(data_size, 1);
outerrmovar1 = zeros(data_size, 1);
outerrmovar2 = zeros(data_size, 1);
outerrmovar = zeros(data_size, 1);

EPS = eps();

%% Post-calc variables
PredErrorsNL = zeros(T, 1);
MSE_NL = zeros(1, 6);

PredErrorsDEU = zeros(T, 1);
MSE_DEU = zeros(1, 6);

%% Initialisation NL
for k = 1:1
    data = infl_NL;
    median_NL(1:data_size, 1) = median(infl_NL);
    
    y = 0;
    T = size(data, 1);
    pf = y/rf;
    rob = 1;
    
    % In-sample performance of the forecasting models
    for h = 1:7
        % Define heuristic predictions
        for i = 3:T
            switch h
                case 1 % Fundamental
                    pexp(i+1, 1) = pf;
                case 2 % Naive
                    pexp(i+1, 1) = data(i-1, 1);
                case 3 % Anchoring and Adjustment
                    pexp(i+1, 1) = pf/2 + 1.5 * data(i-1, 1) - data(i-2, 1);
                case 4 % Adaptive
                    w = 0.65;
                    pexp(2, 1) = data(1, 1);
                    pexp(3, 1) = w * data(1, 1) + (1 - w) * pexp(2, 1);
                    pexp(i + 1, 1) = w * data(i-1, 1) + (1 -w) * pexp(i ,1);
                case 5 % Weak Trend-following
                    gamma = 0.4;
                    pexp(i+1, 1) = data(i-1, 1) + gamma * (data(i-1, 1) - data(i-2, 1));
                case 6 % Strong Trend-following
                    gamma = 1.3;
                    pexp(i+1, 1) = data(i-1, 1) + gamma * (data(i-1, 1) - data(i-2, 1));
                case 7 % Learning Anchoring and Adjustment
                    ave = mean(data(1:i-1),1);
                    pexp(i+1, 1) = 0.5 * ave + 1.5 * data(i-1, 1) - data(i-2, 1);
            end
            
            pexp(i+1, 1) = max(lower, min(upper, pexp(i+1, 1)));
            
            % Find predicted price and prediction error
            price(i, k) = ((1 - robots(i, k)) * pexp(i+1, 1) + robots(i, k) * pf + y + noise(i, k)) / (1 + rf);
            error(i, k) = price(i, k) - data(i, 1);
        end
        
        mse(h, k) = sum(error(5:T, k).^2)/(T - 4);
        
    end
    
    Output_NL = HSMRoot(T, [0.4 0.7 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
    
    NL_EtaTrial0 = HSMRoot(T, [0.4 0.0 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
    NL_EtaTrial1 = HSMRoot(T, [0.4 0.1 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0); 
    NL_EtaTrial2 = HSMRoot(T, [0.4 0.2 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
    NL_EtaTrial3 = HSMRoot(T, [0.4 0.3 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
    NL_EtaTrial4 = HSMRoot(T, [0.4 0.4 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
    NL_EtaTrial5 = HSMRoot(T, [0.4 0.5 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
    NL_EtaTrial6 = HSMRoot(T, [0.4 0.6 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
    NL_EtaTrial7 = HSMRoot(T, [0.4 0.7 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
    NL_EtaTrial8 = HSMRoot(T, [0.4 0.8 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
    NL_EtaTrial9 = HSMRoot(T, [0.4 0.9 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
    NL_EtaTrial10 = HSMRoot(T, [0.4 1.0 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
end

%% Initialisation DE
for k = 1:1
    data = infl_DEU;
    median_DEU(1:data_size, 1) = median(infl_DEU);
    
    y = 0;
    T = size(data, 1);
    pf = y/rf;
    rob = 1;
    
    % In-sample performance of the forecasting models
    for h = 1:7
        % Define heuristic predictions
        for i = 3:T
            switch h
                case 1 % Fundamental
                    pexp(i+1, 1) = pf;
                case 2 % Naive
                    pexp(i+1, 1) = data(i-1, 1);
                case 3 % Anchoring and Adjustment
                    pexp(i+1, 1) = pf/2 + 1.5 * data(i-1, 1) - data(i-2, 1);
                case 4 % Adaptive
                    w = 0.65;
                    pexp(2, 1) = data(1, 1);
                    pexp(3, 1) = w * data(1, 1) + (1 - w) * pexp(2, 1);
                    pexp(i + 1, 1) = w * data(i-1, 1) + (1 -w) * pexp(i ,1);
                case 5 % Weak Trend-following
                    gamma = 0.4;
                    pexp(i+1, 1) = data(i-1, 1) + gamma * (data(i-1, 1) - data(i-2, 1));
                case 6 % Strong Trend-following
                    gamma = 1.3;
                    pexp(i+1, 1) = data(i-1, 1) + gamma * (data(i-1, 1) - data(i-2, 1));
                case 7 % Learning Anchoring and Adjustment
                    ave = mean(data(1:i-1),1);
                    pexp(i+1, 1) = 0.5 * ave + 1.5 * data(i-1, 1) - data(i-2, 1);
            end
            
            pexp(i+1, 1) = max(lower, min(upper, pexp(i+1, 1)));
            
            % Find predicted price and prediction error
            price(i, k) = ((1 - robots(i, k)) * pexp(i+1, 1) + robots(i, k) * pf + y + noise(i, k)) / (1 + rf);
            error(i, k) = price(i, k) - data(i, 1);
        end
        
        mse(h, k) = sum(error(5:T, k).^2)/(T - 4);
        
    end
    
    Output_DEU = HSMRoot(T, [0.4 0.7 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
    
    DEU_EtaTrial0 = HSMRoot(T, [0.4 0.0 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0); 
    DEU_EtaTrial1 = HSMRoot(T, [0.4 0.1 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0); 
    DEU_EtaTrial2 = HSMRoot(T, [0.4 0.2 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
    DEU_EtaTrial3 = HSMRoot(T, [0.4 0.3 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
    DEU_EtaTrial4 = HSMRoot(T, [0.4 0.4 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
    DEU_EtaTrial5 = HSMRoot(T, [0.4 0.5 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
    DEU_EtaTrial6 = HSMRoot(T, [0.4 0.6 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
    DEU_EtaTrial7 = HSMRoot(T, [0.4 0.7 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
    DEU_EtaTrial8 = HSMRoot(T, [0.4 0.8 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
    DEU_EtaTrial9 = HSMRoot(T, [0.4 0.9 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
    DEU_EtaTrial10 = HSMRoot(T, [0.4 1.0 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
end

%% Initialisation Rednoise
for k = 1:1
    data = infl_unstable;
    
    y = 0;
    T = size(data, 1);
    pf = y/rf;
    rob = 1;
    
    % In-sample performance of the forecasting models
    for h = 1:7
        % Define heuristic predictions
        for i = 3:T
            switch h
                case 1 % Fundamental
                    pexp(i+1, 1) = pf;
                case 2 % Naive
                    pexp(i+1, 1) = data(i-1, 1);
                case 3 % Anchoring and Adjustment
                    pexp(i+1, 1) = pf/2 + 1.5 * data(i-1, 1) - data(i-2, 1);
                case 4 % Adaptive
                    w = 0.65;
                    pexp(2, 1) = data(1, 1);
                    pexp(3, 1) = w * data(1, 1) + (1 - w) * pexp(2, 1);
                    pexp(i + 1, 1) = w * data(i-1, 1) + (1 -w) * pexp(i ,1);
                case 5 % Weak Trend-following
                    gamma = 0.4;
                    pexp(i+1, 1) = data(i-1, 1) + gamma * (data(i-1, 1) - data(i-2, 1));
                case 6 % Strong Trend-following
                    gamma = 1.3;
                    pexp(i+1, 1) = data(i-1, 1) + gamma * (data(i-1, 1) - data(i-2, 1));
                case 7 % Learning Anchoring and Adjustment
                    ave = mean(data(1:i-1),1);
                    pexp(i+1, 1) = 0.5 * ave + 1.5 * data(i-1, 1) - data(i-2, 1);
            end
            
            pexp(i+1, 1) = max(lower, min(upper, pexp(i+1, 1)));
            
            % Find predicted price and prediction error
            price(i, k) = ((1 - robots(i, k)) * pexp(i+1, 1) + robots(i, k) * pf + y + noise(i, k)) / (1 + rf);
            error(i, k) = price(i, k) - data(i, 1);
        end
        
        mse(h, k) = sum(error(5:T, k).^2)/(T - 4);
        
    end
    
    Output_Unstable = HSMRoot(T, [0.4 0.7 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
    
%     DEU_EtaTrial1 = HSMRoot(T, [0.4 0.1 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0); 
%     DEU_EtaTrial2 = HSMRoot(T, [0.4 0.2 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
%     DEU_EtaTrial3 = HSMRoot(T, [0.4 0.3 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
%     DEU_EtaTrial4 = HSMRoot(T, [0.4 0.4 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
%     DEU_EtaTrial5 = HSMRoot(T, [0.4 0.5 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
%     DEU_EtaTrial6 = HSMRoot(T, [0.4 0.6 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
%     DEU_EtaTrial7 = HSMRoot(T, [0.4 0.7 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
%     DEU_EtaTrial8 = HSMRoot(T, [0.4 0.8 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
%     DEU_EtaTrial9 = HSMRoot(T, [0.4 0.9 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
%     DEU_EtaTrial10 = HSMRoot(T, [0.4 0.10 0.9], data, [rf y lower upper], noise(:,k), [1 1 1 1 0 0], 0);
end

%% Post calculations
for j = 1:6
    PredErrorsNL(:, j) = Output_NL.predictions(data_size, j) - infl_NL(:, 1);
    MSE_NL(1, j) = mean(PredErrorsNL(:, j).^2);
end

for j = 1:6
    PredErrorsDEU(:, j) = Output_DEU.predictions(data_size, j) - infl_DEU(:, 1);
    MSE_DEU(1, j) = mean(PredErrorsDEU(:, j).^2);
end

ABSPredErrorsNL = abs(PredErrorsNL);
ABSPredErrorsDEU = abs(PredErrorsDEU);

% Memory parameter Netherlands
figure('Name', 'The Netherlands: Memory Parameter')
subplot(6, 2, 1)
plot(NL_EtaTrial0.fraction(:, 1:5))
title('\eta = 0.0')
xlabel('Time (months)')

subplot(6, 2, 2)
plot(NL_EtaTrial1.fraction(:, 1:5))
title('\eta = 0.1')
xlabel('Time (months)')

subplot(6, 2, 3)
plot(NL_EtaTrial2.fraction(:, 1:5))
title('\eta = 0.2')
xlabel('Time (months)')

subplot(6, 2, 4)
plot(NL_EtaTrial3.fraction(:, 1:5))
title('\eta = 0.3')
xlabel('Time (months)')

subplot(6, 2, 5)
plot(NL_EtaTrial4.fraction(:, 1:5))
title('\eta = 0.4')
xlabel('Time (months)')

subplot(6, 2, 6)
plot(NL_EtaTrial5.fraction(:, 1:5))
title('\eta = 0.5')
xlabel('Time (months)')

subplot(6, 2, 7)
plot(NL_EtaTrial6.fraction(:, 1:5))
title('\eta = 0.6')
xlabel('Time (months)')

subplot(6, 2, 8)
plot(NL_EtaTrial7.fraction(:, 1:5))
title('\eta = 0.7')
xlabel('Time (months)')

subplot(6, 2, 9)
plot(NL_EtaTrial8.fraction(:, 1:5))
title('\eta = 0.8')
xlabel('Time (months)')

subplot(6, 2, 10)
plot(NL_EtaTrial9.fraction(:, 1:5))
title('\eta = 0.9')
xlabel('Time (months)')

subplot(6, 2, 11)
plot(NL_EtaTrial10.fraction(:, 1:5))
title('\eta = 1.0')
xlabel('Time (months)')

% Memory parameter Germany
figure('Name', 'Germany: Memory Parameter')
subplot(6, 2, 1)
plot(DEU_EtaTrial0.fraction(:, 1:5))
title('\eta = 0.0')
xlabel('Time (months)')

subplot(6, 2, 2)
plot(DEU_EtaTrial1.fraction(:, 1:5))
title('\eta = 0.1')
xlabel('Time (months)')

subplot(6, 2, 3)
plot(DEU_EtaTrial2.fraction(:, 1:5))
title('\eta = 0.2')
xlabel('Time (months)')

subplot(6, 2, 4)
plot(DEU_EtaTrial3.fraction(:, 1:5))
title('\eta = 0.3')
xlabel('Time (months)')

subplot(6, 2, 5)
plot(DEU_EtaTrial4.fraction(:, 1:5))
title('\eta = 0.4')
xlabel('Time (months)')

subplot(6, 2, 6)
plot(DEU_EtaTrial5.fraction(:, 1:5))
title('\eta = 0.5')
xlabel('Time (months)')

subplot(6, 2, 7)
plot(DEU_EtaTrial6.fraction(:, 1:5))
title('\eta = 0.6')
xlabel('Time (months)')

subplot(6, 2, 8)
plot(DEU_EtaTrial7.fraction(:, 1:5))
title('\eta = 0.7')
xlabel('Time (months)')

subplot(6, 2, 9)
plot(DEU_EtaTrial8.fraction(:, 1:5))
title('\eta = 0.8')
xlabel('Time (months)')

subplot(6, 2, 10)
plot(DEU_EtaTrial9.fraction(:, 1:5))
title('\eta = 0.9')
xlabel('Time (months)')

subplot(6, 2, 11)
plot(DEU_EtaTrial10.fraction(:, 1:5))
title('\eta = 1.0')
xlabel('Time (months)')