% test_classifier: This function takes a single record from the
% challenge training set, and uses that record together with the
% accompanying arousal annotations to tune model parameters
% appropriately. You may want to use this function as a basis for your
% own training code.
%
% Written by Mohammad Ghassemi and Benjamin Moody, 2018

function [predictions, n_samples] = data_to_run_classifier_hybrid(header_file_name)
% Read record info from the header file
data = parse_header(header_file_name);

X_te = [];

% collect a list of all the trained models
files = dir(); files = {files.name};
models = find(contains(files,'netM'));

%load all the the data associated with this subject
signals      = load(data.signal_location); signals = signals.val;
fs           = str2num(data.fs);
%n_samples    = str2num(data.n_samples);
sid          = data.subject_id;
signal_names = data.signal_names;

% select the window size and step size we want to use to
% compute features
window_size = 60 * fs;
window_step = 30 * fs;

% find the index of the signals.
eeg1=signals(1,:);eeg2=signals(2,:);eeg3=signals(3,:);eeg4=signals(4,:);eeg5=signals(5,:); eeg6=signals(6,:);eog7=signals(7,:); emg8=signals(8,:);%Sao2=signals(12,:);
abd=signals(9,:);chest=signals(10,:);air=signals(11,:);ecg=signals(13,:);
n_samples=size(eeg1,2);
kk3 = 1:window_step:n_samples-window_size;
kk=size(kk3,2);
k3=1;ind=1;


for jj = 1:window_step:n_samples-window_size
    
    
    pp1=eeg1(jj:jj+window_size);
    pp2=eeg2(jj:jj+window_size);
    pp3=eeg3(jj:jj+window_size);
    pp4=eeg4(jj:jj+window_size);
    pp5=eeg5(jj:jj+window_size);
    pp6=eeg6(jj:jj+window_size);
    pp7=eog7(jj:jj+window_size);
    pp8=emg8(jj:jj+window_size);
    pp9=abd(jj:jj+window_size);
    pp10=chest(jj:jj+window_size);
    pp11=air(jj:jj+window_size);
    %pp12=Sao2(jj:jj+window_size);
    pp13=ecg(jj:jj+window_size);
    
%     psao2=downsample(pp12,46)';
%     psao2=psao2([1:258],1)';
    
    
    
    pp=[pp1;pp2;pp3;pp4;pp5;pp6;pp7;pp8;pp9;pp10;pp11;pp13];
    %ppd=[psao2];
    
    %ppd1(:,k3)={ppd};
    
    seg(:,k3)={pp};
    k3=k3+1;
    
end
feat1 = cellfun(@(x)instfreq(x(1,:),200)',seg,'UniformOutput',false);
feat2 = cellfun(@(x)instfreq(x(2,:),200)',seg,'UniformOutput',false);
feat3 = cellfun(@(x)instfreq(x(3,:),200)',seg,'UniformOutput',false);
feat4 = cellfun(@(x)instfreq(x(4,:),200)',seg,'UniformOutput',false);
feat5 = cellfun(@(x)instfreq(x(5,:),200)',seg,'UniformOutput',false);
feat6 = cellfun(@(x)instfreq(x(6,:),200)',seg,'UniformOutput',false);
feat7 = cellfun(@(x)instfreq(x(7,:),200)',seg,'UniformOutput',false);
feat8 = cellfun(@(x)instfreq(x(8,:),200)',seg,'UniformOutput',false);
feat9 = cellfun(@(x)instfreq(x(9,:),200)',seg,'UniformOutput',false);
feat10 = cellfun(@(x)instfreq(x(10,:),200)',seg,'UniformOutput',false);
feat11 = cellfun(@(x)instfreq(x(11,:),200)',seg,'UniformOutput',false);
feat12 = cellfun(@(x)instfreq(x(12,:),200)',seg,'UniformOutput',false);

feat13 = cellfun(@(x)pentropy(x(1,:),200)',seg,'UniformOutput',false);
feat14 = cellfun(@(x)pentropy(x(2,:),200)',seg,'UniformOutput',false);
feat15 = cellfun(@(x)pentropy(x(3,:),200)',seg,'UniformOutput',false);
feat16 = cellfun(@(x)pentropy(x(4,:),200)',seg,'UniformOutput',false);
feat17 = cellfun(@(x)pentropy(x(5,:),200)',seg,'UniformOutput',false);
feat18 = cellfun(@(x)pentropy(x(6,:),200)',seg,'UniformOutput',false);
feat19 = cellfun(@(x)pentropy(x(7,:),200)',seg,'UniformOutput',false);
feat20 = cellfun(@(x)pentropy(x(8,:),200)',seg,'UniformOutput',false);
feat21 = cellfun(@(x)pentropy(x(9,:),200)',seg,'UniformOutput',false);
feat22 = cellfun(@(x)pentropy(x(10,:),200)',seg,'UniformOutput',false);
feat23 = cellfun(@(x)pentropy(x(11,:),200)',seg,'UniformOutput',false);
feat24 = cellfun(@(x)pentropy(x(12,:),200)',seg,'UniformOutput',false);

feat = cellfun(@(x,y,z,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u)[x;y;z;a;b;c;d;e;f;g;h;i;j;k;l;m;n;o;p;q;r;s;t;u],feat1,feat2,feat3,feat4,feat5,feat6,feat7,feat8,feat9,feat10,feat11,feat12,feat13,feat14,feat15,feat16,feat17,feat18,feat19,feat20,feat21,feat22,feat23,feat24,'UniformOutput',false);
XV = [feat{:}];
XV(isnan(XV))=0;
mu = mean(XV,2);
sg = std(XV,[],2);

for i=1:kk %(length(XV)/258)
    a=258*(i-1)+1;
    b=258*i;
    Newfeat1{i} = XV(:,a:b);
    
end
NewfeaTest=Newfeat1';
feat = NewfeaTest;
feat = cellfun(@(x)(x-mu)./sg,feat,'UniformOutput',false);
X_te=feat;
% generate the probability vectors
        display('Generating Scores')
        for k = 1:length(models)
                % loading model
                load(files{models(k)});

                % generate the probability vectors
                pred_short = classify(netM,X_te);
                pred_short2= grp2idx(pred_short);
                pred_short2(pred_short2==2)=0;
                pred_short=pred_short2;
                
                pred1(:,k)=pred_short;
        end
        predictions = pred1;