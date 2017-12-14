%% The Netherlands

% Import the data
[~, ~, raw] = xlsread('C:\Users\Laurens\Dropbox\University\MSc\Thesis\Scripts\InflationBankWashington.xlsx','ATS','D2:D360');
raw(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),raw)) = {''};

% Replace non-numeric cells with NaN
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
raw(R) = {NaN}; % Replace non-numeric cells

% Create output variable
infl_NL = reshape([raw{:}],size(raw));

% Clear temporary variables
clearvars raw R;

%% Germany

% Import the data
[~, ~, raw] = xlsread('C:\Users\Laurens\Dropbox\University\MSc\Thesis\Scripts\InflationBankWashington.xlsx','ATS','C2:C360');
raw(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),raw)) = {''};

% Replace non-numeric cells with NaN
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
raw(R) = {NaN}; % Replace non-numeric cells

% Create output variable
infl_DEU = reshape([raw{:}],size(raw));

% Clear temporary variables
clearvars raw R;