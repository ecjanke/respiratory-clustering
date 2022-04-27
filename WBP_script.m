clear 
clc
data = TDTbin2mat('/Users/emmajanke/Documents/Code (Matlab)/TDTMatlabSDK/Pleth/1130305_113306_1-210504-123619');% change name of file

STREAM_STORE1 = 'Res1';
respiratoryTrace = data.streams.(STREAM_STORE1).data;
Fs = 300;

time = [1:length(respiratoryTrace)]/Fs;
dataType = 'rodentAirflow';
plot(time,respiratoryTrace,'-k')

new_resp = smooth(time,respiratoryTrace,15);
plot(time,new_resp)
%% create BM object for whole trace
BM = breathmetrics(new_resp,Fs,dataType);

verbose=1; 

baselineCorrectionMethod = 'sliding'; 
zScore=1;                                %z-scoring data to compare across animals
BM.correctRespirationToBaseline(baselineCorrectionMethod, zScore, verbose)

plot(BM.time,BM.baselineCorrectedRespiration,'-k');

T1 = [];            %vector behavior initiation timepoints  (s)
T2 = [];            %vector behavior termination timepoints (s)

%% Restart BM for behavioral timept -> do NOT reprocess data
T1 = T1*Fs;         %behav on
T2 = T2*Fs;         %behav off
  
N = numel(T1);
C = cell(1,N);
tmp = [];

for k=1:N;          %create structure of objects - each corresponding to one bout
    tmp = BM.baselineCorrectedRespiration(1,T1(k):T2(k));
    C{k} = breathmetrics(tmp,Fs,dataType);
end

for k=1:N;
    bouts{k} = BM.baselineCorrectedRespiration(1,T1(k):T2(k));
end


S = [C{:}];
for i=1:N;
    S(i).baselineCorrectedRespiration = S(i).rawRespiration;     %overwrite so not baseline correcting 2x
    i=i+1;
end

simplify=0;
verbose=1;
for i=1:N;          %calculate features on each bout
    S(i).findExtrema(simplify,verbose);
    S(i).findOnsetsAndPauses(verbose);
    S(i).findInhaleAndExhaleOffsets(S(i));
    findBreathAndPauseDurations(S(i));
    S(i).findInhaleAndExhaleVolumes(verbose);
    S(i).getSecondaryFeatures(S(i));
    i=i+1;
end

for i=1:N;         %extract secondary features
    vals{i} = values(S(i).secondaryFeatures);
    i=i+1;
end

final = ones([N,24]);   %secondary features --> matrix
for i=1:N
     final(i,1:24) = cell2mat(vals{1,i});
end

%%  data visualization
% shows labeling of each bout by breathmetrics
% press any key in command window to continue to plotting next bout
for i=1:length(S)
    pause;
    PLOT_LIMITS = 1:length(S(i).baselineCorrectedRespiration);
    inhalePeaksWithinPlotlimits = find(S(i).inhalePeaks >= min(PLOT_LIMITS) ...
        & S(i).inhalePeaks < max(PLOT_LIMITS));
    exhaleTroughsWithinPlotlimits = find(S(i).exhaleTroughs >= ...
        min(PLOT_LIMITS) & S(i).exhaleTroughs < max(PLOT_LIMITS));
    inhaleOnsetsWithinPlotLimits = find(S(i).inhaleOnsets >= min(PLOT_LIMITS) ...
        & S(i).inhaleOnsets<max(PLOT_LIMITS));
    exhaleOnsetsWithinPlotLimits = find(S(i).exhaleOnsets >= min(PLOT_LIMITS) ...
        & S(i).exhaleOnsets<max(PLOT_LIMITS));
    inhalePausesWithinPlotLimits = find(S(i).inhalePauseOnsets >= ...
        min(PLOT_LIMITS) & S(i).inhalePauseOnsets < max(PLOT_LIMITS));
    exhalePausesWithinPlotLimits = find(S(i).exhalePauseOnsets >= ...
        min(PLOT_LIMITS) & S(i).exhalePauseOnsets<max(PLOT_LIMITS));
    inhaleOffsetsWithinPlotLimits = find(S(i).inhaleOffsets >= ...
        min(PLOT_LIMITS) & S(i).inhaleOffsets < max(PLOT_LIMITS));
    exhaleOffsetsWithinPlotLimits = find(S(i).exhaleOffsets >= ...
        min(PLOT_LIMITS) & S(i).exhaleOffsets < max(PLOT_LIMITS));
   

    figure; hold all;
    re=plot(S(i).time(PLOT_LIMITS), ...
        S(i).baselineCorrectedRespiration(PLOT_LIMITS),'k-');
    peak=scatter(S(i).time(S(i).inhalePeaks(inhalePeaksWithinPlotlimits)), ...
        S(i).peakInspiratoryFlows(inhalePeaksWithinPlotlimits),'bo','filled'); 
    tr=scatter(S(i).time(S(i).exhaleTroughs(exhaleTroughsWithinPlotlimits)), ...
        S(i).troughExpiratoryFlows(exhaleTroughsWithinPlotlimits), ...
        'ro', 'filled'); 
    io=scatter(S(i).time(S(i).inhaleOnsets(inhaleOnsetsWithinPlotLimits)), ...
        S(i).baselineCorrectedRespiration( ...
        S(i).inhaleOnsets(inhaleOnsetsWithinPlotLimits)),'mo','filled');
    eo=scatter(S(i).time(S(i).exhaleOnsets(exhaleOnsetsWithinPlotLimits)), ...
        S(i).baselineCorrectedRespiration(S(i).exhaleOnsets( ...
        exhaleOnsetsWithinPlotLimits)), 'co', 'filled');
    ip=scatter(S(i).time(S(i).inhalePauseOnsets(inhalePausesWithinPlotLimits)) ...
        ,S(i).baselineCorrectedRespiration(S(i).inhalePauseOnsets( ...
        inhalePausesWithinPlotLimits)), 'go', 'filled');
    ep=scatter(S(i).time(S(i).exhalePauseOnsets(exhalePausesWithinPlotLimits)), ...
        S(i).baselineCorrectedRespiration(S(i).exhalePauseOnsets( ...
        exhalePausesWithinPlotLimits)), 'yo', 'filled');
    legendText={
        'Baseline Corrected Respiration';
        'Inhale Peaks';
        'Exhale Troughs';
        'Inhale Onsets';
        'Exhale Onsets';
        'Inhale Pause Onsets';
        'Exhale Pause Onsets'
        };
    legend([re,peak,tr,io,eo,ip,ep],legendText);
    xlabel('Time (seconds)');
    ylabel('Respiratory Flow');

    figure; hold all;
    re=plot(S(i).time(PLOT_LIMITS), ...
        S(i).baselineCorrectedRespiration(PLOT_LIMITS),'k-');
    peak=scatter(S(i).time(S(i).inhalePeaks(inhalePeaksWithinPlotlimits)), ...
        S(i).peakInspiratoryFlows(inhalePeaksWithinPlotlimits),'bo','filled'); 
    tr=scatter(S(i).time(S(i).exhaleTroughs(exhaleTroughsWithinPlotlimits)), ...
        S(i).troughExpiratoryFlows(exhaleTroughsWithinPlotlimits), ...
        'ro', 'filled'); 
    io=scatter(S(i).time(S(i).inhaleOnsets(inhaleOnsetsWithinPlotLimits)), ...
        S(i).baselineCorrectedRespiration(S(i).inhaleOnsets( ...
        inhaleOnsetsWithinPlotLimits)), 'co', 'filled');
    eo=scatter(S(i).time(S(i).exhaleOnsets(exhaleOnsetsWithinPlotLimits)), ...
        S(i).baselineCorrectedRespiration(S(i).exhaleOnsets( ...
        exhaleOnsetsWithinPlotLimits)),'mo','filled');
    ip=scatter(S(i).time(S(i).inhaleOffsets(inhaleOffsetsWithinPlotLimits)), ...
        S(i).baselineCorrectedRespiration(S(i).inhaleOffsets( ...
        inhaleOffsetsWithinPlotLimits)),'yo','filled');
    ep=scatter(S(i).time(S(i).exhaleOffsets(exhaleOffsetsWithinPlotLimits)), ...
        S(i).baselineCorrectedRespiration(S(i).exhaleOffsets( ...
        exhaleOffsetsWithinPlotLimits)),'go','filled');
    legendText = {
        'Baseline Corrected Respiration';
        'Inhale Peaks';
        'Exhale Troughs';
        'Inhale Onsets';
        'Exhale Onsets';
        'Inhale Offsets';
        'Exhale Offsets'};
    legend([re,peak,tr,io,eo,ip,ep],legendText);
    xlabel('Time (seconds)');
    ylabel('Respiratory Flow');   
end

%%
%import [m,n] matrix for each behavior, m:bouts, n:24 secondary features from Bm

Bm_data = [];

Bm_data = vertcat(PlethSniff,PlethPSA,PlethGroom,PlethFreeze,PlethQui);    %compile behaviors

done_Z = Bm_data(:,9);                         %inspiratory flow already z-scored

Bm = Bm_data(:,[4,11,12,13,18,20]);            %six parameters to z-score

DIM = 1;
Z_bm = zscore(Bm,0,DIM);
TOT_Z = horzcat(Z_bm,done_Z); 
%% Conduct WBP PCA
[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU_PCA] = pca(TOT_Z);
exp = EXPLAINED';

figure
plot3(SCORE(:,1),SCORE(:,2),SCORE(:,3),'bo');
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
legend('Breathing bout');
title('PCA 7 Parameters');

%% WBP KNN
%import compiled 'training' matrix of 7 breathmetrics parameters

done_Z = training(:,2);                 %inspiratory flow: parameter already z-scored

Bm = training(:,[1,3:7]);               %parameters to z-score

DIM = 1;
[Z_bm, mu,sigma] = zscore(Bm,0,DIM);    %z-score
TOT_Z = horzcat(Z_bm,done_Z(:,:));

[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU_PCA] = pca(TOT_Z);          %PCA on training data
%% Project test data --> training PC

% import test matrix of 7 breathmetrics parameters into 'test_Bm' in same
% format as training

%z-scored test data
Z_test = test_Bm(:,[1,3:7]);
Z_test = Z_test - mu;
Z_test = Z_test./sigma;
Z_test = horzcat(Z_test,test_Bm(:,2));       

 %project test to PC
center_test = Z_test-MU_PCA;
PC_test = center_test/COEFF';               

%% build KNN
% load training data table with label, PC scores, weights for each training
% bout

rng = 10;
KNN_fin = fitcknn(Tbl(:,1:8),'Label','W',Tbl.W); 

for i=1:35                                      %cross validation & assess loss for dif number of neighbors
    KNN_fin.NumNeighbors = i;
    CV_KNN_fin = crossval(KNN_fin); 
    kloss = kfoldLoss(CV_KNN_fin);  
    loss(i) = kloss;                 
    i=i+1;
end

loss = loss';

%% adjust number neighbors & predict
KNN_fin.NumNeighbors =6;              %chosen from loss

[label_6, score_6, cost_6] = predict(KNN_fin,PC_test);     
%% create shuffled training set
%w_p = WBP weights (inverse of n in each behavior class)

y = w_p(randperm(numel(w_p)));

label={};
for i=1:278                          %create label list based on shuffled weight
    if y(i) == 0.02326
        label{i,1} = 'Sniff';
    elseif y(i)==0.03226
        label{i,1} = 'PSA';
    elseif y(i)==0.02632
        label{i,1} = 'Groom';
    elseif y(i)==0.01031
        label{i,1} = 'Qui';
    elseif y(i)==0.01449
        label{i,1} = 'Freeze';
    end
end

%% build shuffled KNN & predict
% Tbl is identical to accurate KNN, replace labels with shuffled labels &
% weights

KNN_shuf = fitcknn(Tbl(:,1:8),'Label','W',Tbl.W);

KNN_shuf.NumNeighbors = 6;
[label_sh, score_sh, cost_sh] = predict(KNN_shuf,PC_test);