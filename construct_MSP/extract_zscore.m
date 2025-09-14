clc, clear, close all

dataname = 'SV';

% load enigma summary statistics
enigma = load('../data/enigma_stats_adult.mat');

switch dataname
    case 'SV'
        d_subvol = enigma.cohensd_SUBVOL;
        % load z score
        zdata1 = readtable('../centileBrain/SV_female/zscore_SubcorticalVolume_female.csv');
        zdata2 = readtable('../centileBrain/SV_male/zscore_SubcorticalVolume_male.csv');
    case 'CT'
        d_subvol = enigma.cohensd_CT;
        zdata1 = readtable('../centileBrain/CT_female/zscore_CorticalThickness_female.csv');
        zdata2 = readtable('../centileBrain/CT_male/zscore_CorticalThickness_male.csv');
    case 'SA'
        d_subvol = enigma.cohensd_SA;
        zdata1 = readtable('../centileBrain/SA_female/zscore_SurfaceArea_female.csv');
        zdata2 = readtable('../centileBrain/SA_male/zscore_SurfaceArea_male.csv');
    otherwise
        error('Invalid dataname.');
end

zdata = cat(1, zdata1, zdata2);
zval = table2array(zdata);

if strcmp(dataname, 'SV')
    zval = zval(:, [1:2:13, 2:2:14]);   %要换顺序o(╥﹏╥)o
end

% get group label
person1 = readtable('../centileBrain/data_centileBrain_subcort_female.xlsx');
person2 = readtable('../centileBrain/data_centileBrain_subcort_male.xlsx');
person = cat(1, person1, person2);
data = load('../data/ndyx_VOL_SA_CT_SUBVOL_COV.mat');
[~, J] = ismember(person.subjects, data.subjects);

% 要保存的其他信息
group = data.group(J, :);
name = data.subjects(J, :);
rate = data.rate(J, :);
t1panss = data.t1panss(J, :);
t2panss = data.t2panss(J, :);
diffpanss = t1panss - t2panss;
pos = data.t2panss_pos(J, :);
neg = data.t2panss_neg(J, :);
cog = data.t2panss_cog(J, :);
disease = enigma.diseaseDescriptions;
regionDescriptions = data.regionDescriptions;
subcortexDescriptions = data.subcortexDescriptions;


save(strcat('../similarity_ndyx/zval_', dataname, '.mat'),"zval","group","name","rate", "t1panss", "t2panss", "diffpanss", "disease", "pos", "cog", "neg","subcortexDescriptions", "regionDescriptions", "-mat");


