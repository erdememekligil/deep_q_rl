clc; clear; close all;
%%
init_game_name_map
full_data = {};
fid = fopen('D:\dev\projects\deep_q_rl\deep_q_rl\seq_cluster_full.txt');
tline = fgetl(fid);
i=1;
while ischar(tline)
    spl = strsplit(tline, '\t');
    full_data{i,1} = name_map(spl{1});
    for j=2:length(spl),
        full_data{i,j} = str2num(spl{j});
    end
    
    tline = fgetl(fid);
    i = i + 1;
end
fclose(fid);
%cell2mat(full_data(3,2:end))
%dtw(cell2mat(full_data(1,2:end))', cell2mat(full_data(3,2:end))')
k = 1;
Y = [];
for i=1:size(full_data, 1)
    for j=i+1:size(full_data, 1)
        Y(k) = dtw(cell2mat(full_data(i,2:end))', cell2mat(full_data(j,2:end))');
        k = k + 1;
    end
end

labels = full_data(1:end,1);
linkages = {'single', 'average', 'complete'};

for j=1:length(linkages)
    distance = 'dtw';
    linkage_method = linkages{j};
    h = figure;
    Z = linkage(Y, linkage_method);
    dendrogram(Z,0, 'labels', labels)

    set(gca,'XTickLabelRotation',90)
    ylabel('Distance')
    xlabel('Games')
    %title(['Results for DTW distance and ' linkage_method ' linkage'])

    x0=500;
    y0=000;
    width=1050;
    height=600;
    set(gcf,'units','points','position',[x0,y0,width,height])

    pos = get(gca, 'Position');
    pos(1) = 0.055;
    pos(3) = 0.9;
    set(gca, 'Position', pos)

    set(h,'Units','Inches');
    pos = get(h,'Position');
    set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
    set(gca,'FontSize',12) % Creates an axes and sets its FontSize to 18
    
    print(h,['dendrograms/dendogram_' distance '_' linkage_method '.pdf'],'-dpdf','-r0')
    saveas(gcf,['dendrograms/dendogram_' distance '_' linkage_method '.png'])
end
%%
T = cluster(Z,'maxclust',3);
temp = labels;
%T = cluster(Z,'cutoff',0.55);
for i = 1:size(labels,1)
    temp{i,2} = reverse_name_map(temp{i,1});
    temp{i,3} = T(i);
end
temp = sortrows(temp, 3);
%%
labels = {'gravitar'
'road_runner'
'krull'
'assault'
'zaxxon'
'star_gunner'
'bank_heist'
'wizard_of_wor'
'asterix'
'Fishing_Derby'
'space_invaders'
'kung_fu_master'
'up_n_down'
'boxing'
'seaquest'
'freeway'
'crazy_climber'
'demon_attack'
'Tutankham'
'Bowling'
'ice_hockey'
'hero'
'private_eye'
'time_pilot'
'video_pinball'
'double_dunk'
'amidar'
'ms_pacman'
'chopper_command'
'Enduro'
'atlantis'
'frostbite'
'pong'
'kangaroo'
'alien'
'battle_zone'
'riverraid'
'tennis'
'breakout'
'centipede'
'robotank'
'name_this_game'
'gopher'
'venture'
'Qbert'};

%D:\dev\projects\deep_q_rl\deep_q_rl\seq_cluster_full.txt

h = figure;

Y = pdist(X);
%Y = pdist(X, 'hamming');
Z = linkage(Y);
dendrogram(Z,0, 'labels', labels)

set(gca,'XTickLabelRotation',90)
ylabel('Distance')
xlabel('Games')

pos = get(gca, 'Position');
pos(1) = 0.055;
pos(3) = 0.9;
set(gca, 'Position', pos)

set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
set(gca,'FontSize',12) % Creates an axes and sets its FontSize to 18

print(h,'dendogram.pdf','-dpdf','-r0')


T = cluster(Z,'maxclust',7);
%T = cluster(Z,'cutoff',0.55);
for i = 1:size(X,1)
labels{i,2} = T(i);
end