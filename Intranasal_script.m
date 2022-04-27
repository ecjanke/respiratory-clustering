%% Load .abf file
clear
clc

%loading data

respiratorydata = abfload('10_29 test EJ8_0000.abf');

respiratoryTrace = respiratorydata(:,1);
LED = respiratorydata(:,2);

Fs = 2000;
time = (1:length(respiratoryTrace))/Fs; %create time vector

dataType = 'rodentAirflow';
plot(time,respiratoryTrace,'-k');

%% OR Load data: TDT system
clear 
clc

data = TDTbin2mat('/Users/emmajanke/Documents/Code (Matlab)/TDTMatlabSDK/WT11_2-210525-090659');% change name of file

STREAM_STORE1 = 'LFP1';
respiratoryTrace = data.streams.(STREAM_STORE1).data;

Fs = 1017;
time = [1:length(respiratoryTrace)]/Fs;

dataType = 'rodentAirflow';
plot(time,respiratoryTrace,'-k')

%% create BM object whole trace
BM = breathmetrics(respiratoryTrace,Fs,dataType);

verbose=1; 

baselineCorrectionMethod = 'sliding'; 
zScore=1;                                %z-scoring data to compare across animals
BM.correctRespirationToBaseline(baselineCorrectionMethod, zScore, verbose)

plot(BM.time,BM.baselineCorrectedRespiration,'-k');

T1 = []; %vector: behavior initiation timepoints in sec
T2 = []; %vector: behavior termination timepoints in sec

%% Restart BM for behavioral timept -> do NOT reprocess data
T1 = T1*Fs;         %behavior on
T2 = T2*Fs;         %behavior off
  
N = numel(T1);
C = cell(1,N);
tmp = [];

for k=1:N;                                                        %create structure of Bm objects - one per behavior bout run on baseline corrected resp
    tmp = BM.baselineCorrectedRespiration(1,T1(k):T2(k));
    C{k} = breathmetrics(tmp,Fs,dataType);
end

for k=1:N;                                                        %structure of baseline corrected resp traces
    bouts{k} = BM.baselineCorrectedRespiration(1,T1(k):T2(k));
end
%%
S = [C{:}];
for i=1:N;
    S(i).baselineCorrectedRespiration = S(i).rawRespiration;      %overwrite baseline corrected for feature calculation (already performed above)
    i=i+1;
end

simplify=0;
verbose=1;
for i=1:N;                                                        %calculate features on each bout
    S(i).findExtrema(simplify,verbose);                           %Bm uses inverse orientation of airflow respiration (need to relabel inhale features as exhale features and vice versa)
    S(i).findOnsetsAndPauses(verbose);
    S(i).findInhaleAndExhaleOffsets(S(i));
    findBreathAndPauseDurations(S(i));
    S(i).findInhaleAndExhaleVolumes(verbose);
    S(i).getSecondaryFeatures(S(i));
    i=i+1;
end

for i=1:N;                                                        %extract secondary features
    vals{i} = values(S(i).secondaryFeatures);
    i=i+1;
end

final = ones([N,24]);                                             %final secondary features --> matrix                                                             
for i=1:N
     final(i,1:24) = cell2mat(vals{1,i});
end
%%  data visualization
% shows labeling of each bout
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
        'Exhale Peaks';
        'Inhale Troughs';
        'Exhale Onsets';
        'Inhale Onsets';
        'Exhale Pause Onsets';
        'Inhale Pause Onsets'
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
        'Exhale Peaks';
        'Inhale Troughs';
        'Exhale Onsets';
        'Inhale Onsets';
        'Exhale Offsets';
        'Inhale Offsets'};
    legend([re,peak,tr,io,eo,ip,ep],legendText);
    xlabel('Time (seconds)');
    ylabel('Respiratory Flow');   
end
%% Compile Intranasal data: PCA & clustering

Bm_data = [];
Bm_data = vertcat(app,groom,NREM,2MT,Nonfrz2MT,Ret,NonfrzRet,StrugTST,AmmTST,StrugRes,ImmRes); %all data --> [m,n] matrix 
                                                                                               % m: behavior bouts; n:24 Bm secondary features                                                                                             % n: 24 Bm secondary features

Bm = Bm_data(:,[1,2,4,5,11:13,16,18,20,23,24]);                                                % extract 12 features for z-score
done_Z = Bm_data(:,8);                                                                         % extract inspiratory flow (already z-scored)

DIM = 1;
[Z_bm, mu, sigma] = zscore(Bm,0,DIM);                 %z-score

%% create correlation matrix: 20 parameters   %4 excluded parameters frequently included NaN

Bm_lim = Bm_data(:,[1:14,16,18,20,22:24]); 
R2 = corrcoef(Bm_lim);

imagesc(R2); % Display correlation matrix as an image
set(gca, 'XTick', 1:20); % center x-axis ticks on bins
set(gca, 'YTick', 1:20); % center y-axis ticks on bins
title('Parameter Correlation', 'FontSize', 10); % set title

%% Conduct PCA

[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU_PCA] = pca(TOT_Z);
exp = EXPLAINED';
figure
plot3(SCORE(:,1),SCORE(:,2),SCORE(:,3),'bo');
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
legend('Breathing bout');
title('13 Parameters');
%% behavior indices %read in var from excel sheets

app = find(behav_idx==1);
gr = find(behav_idx==2);
NREM = find(behav_idx==3);
pred = find(behav_idx==4);
pred_loco = find(behav_idx==5);
ret = find(behav_idx==6);
ret_loco = find(behav_idx==7);
str_T = find(behav_idx==8);
imm_T = find(behav_idx==9);
str_R = find(behav_idx==10);
imm_R = find(behav_idx==11);

%% PCA plot by behavior

figure
plot3(SCORE(app,1),SCORE(app,2),SCORE(app,3),'.','MarkerSize',10,'Color',[1 0.31 0.023]);      
hold on;
plot3(SCORE(gr,1),SCORE(gr,2),SCORE(gr,3),'*','MarkerSize',10,'Color',[1 0 1]);  
hold on;
plot3(SCORE(NREM,1),SCORE(NREM,2),SCORE(NREM,3),'.','MarkerSize',10,'Color',[1 0.31 0.54]);  
hold on;
plot3(SCORE(pred,1),SCORE(pred,2),SCORE(pred,3),'.','MarkerSize',10,'Color',[0 0.5 0.25]);   
hold on;
plot3(SCORE(ret,1),SCORE(ret,2),SCORE(ret,3),'.','MarkerSize',10,'Color',[0 0.76 1]); 
hold on;
plot3(SCORE(str_T,1),SCORE(str_T,2),SCORE(str_T,3),'*','MarkerSize',10,'Color', [0 .7 .8]);
hold on;
plot3(SCORE(imm_T,1),SCORE(imm_T,2),SCORE(imm_T,3),'.','MarkerSize',10, 'Color','k'); 
hold on;
plot3(SCORE(str_R,1),SCORE(str_R,2),SCORE(str_R,3),'o','MarkerSize',5, 'Color','k');
hold on;
plot3(SCORE(imm_R,1),SCORE(imm_R,2),SCORE(imm_R,3),'o','MarkerSize',5, 'Color','m'); 
hold on;
plot3(SCORE(pred_loco,1),SCORE(pred_loco,2),SCORE(pred_loco,3),'.','MarkerSize',10,'Color',[0 0.2 1]); 
hold on;
plot3(SCORE(ret_loco,1),SCORE(ret_loco,2),SCORE(ret_loco,3),'.','MarkerSize',10,'Color',[0 0.88 0.6]); 
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
legend('Odor investigation','Groom','NREM', '2MT','Retrieval','strTST', 'immTST','strRestraint','immRestraint','2MT loco','Retrieval Loco');
%% k-means

%conduct unbiased clustering off PCA; 3 replicates
k = 4;                                          %k=cluster number
[IDX, C, SUMD, D] = kmeans(SCORE(:,1:3),k);     %SUMD for elbow plot

[IDX1, C1, SUMD1, D1] = kmeans(SCORE(:,1:3),k);

[IDX2,C2,SUMD2,D2] = kmeans(SCORE(:,1:3),k);

c1 = find(IDX==1); %indices of cluster1... 
c2 = find(IDX==2);
c3 = find(IDX==3);
c4 = find(IDX==4);
%% 
%tables for spread of bouts across clusters

ans_app = tabulate(IDX(app));
ans_gr = tabulate(IDX(gr));
ans_NREM = tabulate(IDX(NREM));
ans_pre = tabulate(IDX(pred));
ans_NFP = tabulate(IDX(pred_loco));
ans_ret = tabulate(IDX(ret));
ans_NFR = tabulate(IDX(ret_loco));
ans_immTST = tabulate(IDX(imm_T));
ans_strTST= tabulate(IDX(str_T));
ans_immRes = tabulate(IDX(imm_R));
ans_strRes = tabulate(IDX(str_R));

%% plot as clusters

figure
plot3(SCORE(c1,1),SCORE(c1,2),SCORE(c1,3),'.','MarkerSize',11,'Color',[0.984 0.309 0.023])%;
hold on;
plot3(SCORE(c2,1),SCORE(c2,2),SCORE(c2,3),'.','MarkerSize',11,'Color',[0.109 0.666 .999]);
hold on;
plot3(SCORE(c3,1),SCORE(c2,2),SCORE(c2,3),'.','MarkerSize',11,'Color',[0.553 0.333 0.827]);
hold on;
plot3(SCORE(c4,1),SCORE(c4,2),SCORE(c4,3),'.','MarkerSize',11,'Color',[0.054 0.5 0.486]);
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
legend('High Frequency, High Inspiratory Flow Breathing','Steady Breathing with Even Inhales to Exhales','Brathing with Frequent Long Inhale Pauses','Variable Breathing with Exhale Pauses');
xlim([-4,6]);
ylim([-4,6]);
zlim([-5,8]);

%% KNN
%PCA to build KNN model

%load in training matrix (80% of overall data, 13 relevant BreathMetrics features)
done_Z = training(:,5);                         %features already z-scored

Bm = training(:,[1:4,6:13]);                    %features to z-score

DIM = 1;
[Z_bm, mu,sigma] = zscore(Bm,0,DIM);            %Zscore

TOT_Z = horzcat(Z_bm,done_Z(:,:));              

[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU_PCA] = pca(TOT_Z);          % conduct PCA
%%
%project new to PC intranasal pressure
%load test_BM and zscore

Z_test = test_Bm(:,[1:4,6:13]);                 %import remaining 20% test data in same format as training

%z-score with training parameters
Z_test = Z_test - mu;                           
Z_test = Z_test./sigma;                         
Z_test = horzcat(Z_test,test_Bm(:,5));

%project to training PC space
center_test = Z_test-MU_PCA;            
PC_test = center_test/COEFF';       
%%
%load label from Table
% actual = {};
% actual = actual.Label; (Tablename.var)
rng = 10;

KNN = fitcknn(Tbl(:,1:14),'Label','W',Tbl.W); 

for i=1:35                                      %calculate loss for each number of neighbors
    KNN.NumNeighbors = i;
    CV_KNN_fin = crossval(KNN);                 %cross validate
    kloss = kfoldLoss(CV_KNN_fin);              %assess loss
    loss(i) = kloss;                            %add to vector
    i=i+1;
end


KNN.NumNeighbors=11;                                    % choose lowest loss
[label_11, score_11, cost_11] = predict(KNN,PC_test);   %predict

C = confusionmat(actual,label_11);                    %confusion matrix compare actual to predicted
confusionchart(C)

%% shuffled model

y = w(randperm(numel(w)));         % w: vector of weights in training data

label={};
for i=1:750                         % retroactively assign associated label for shuffled list of weights
    if y(i) == 0.00588
        label{i,1} = 'Active Sniff';
    elseif y(i)==0.02381
        label{i,1} = 'Groom';
    elseif y(i)==0.01515
        label{i,1} = 'NREM';
    elseif y(i)==0.01042
        label{i,1} = '2MT';
    elseif y(i)==0.01351
        label{i,1} = 'Ret';
    elseif y(i)==0.01613
        label{i,1} = 'immTST';
    elseif y(i)==0.016129
        label{i,1} = 'strTST';
    elseif y(i)==0.00980
        label{i,1} = 'immRes';
    elseif y(i)==0.01316
        label{i,1} = 'strRes';
    end
end

%%
KNN_shuf = fitcknn(Tbl(:,1:14),'Label','W',Tbl.W);      %build shuffled model, same PCA training data but with shuffled weight/label assignment

KNN_shuf.NumNeighbors = 11;
[label_sh, score_sh, cost_sh] = predict(KNN_shuf,PC_test);    %predict on testing data
