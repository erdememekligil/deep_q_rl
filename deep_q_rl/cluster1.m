labels = {'Alien'
'Assault'
'Asterix'
'Atlantis'
'Battle Zone'
'Bowling'
'Boxing'
'Breakout'
'Centipede'
'Chopper Command'
'Crazy Climber'
'Demon Attack'
'Double Dunk'
'Enduro'
'Fishing Derby'
'Freeway'
'Frostbite'
'Gopher'
'Ice Hockey'
'Kangaroo'
'Krull'
'Kung-Fu Master'
'Ms. Pacman'
'Name This Game'
'Pong'
'Private Eye'
'Q*bert'
'River Raid'
'Road Runner'
'Robotank'
'Seaquest'
'Space Invaders'
'Star Gunner'
'Tennis'
'Time Pilot'
'Tutankham'
'Up and Down'
'Venture'
'Video pinball'
'Wizard of Wor'
'Zaxxon'
};
%%
labels = {'Gravitar'
'Road Runner'
'Krull'
'Assault'
'Zaxxon'
'Star Gunner'
'Bank Heist'
'Wizard Of Wor'
'Asterix'
'Fishing Derby'
'Space Invaders'
'Kung Fu Master'
'Up N Down'
'Boxing'
'Seaquest'
'Freeway'
'Crazy Climber'
'Demon Attack'
'Tutankham'
'Bowling'
'Ice Hockey'
'Hero'
'Private Eye'
'Time Pilot'
'Video Pinball'
'Double Dunk'
'Amidar'
'Ms Pacman'
'Chopper Command'
'Enduro'
'Atlantis'
'Frostbite'
'Pong'
'Kangaroo'
'Alien'
'Battle Zone'
'Riverraid'
'Tennis'
'Breakout'
'Centipede'
'Robotank'
'Name This Game'
'Gopher'
'Venture'
'Qbert'};
%%
%D:\dev\projects\deep_q_rl\deep_q_rl\seq_cluster.txt

% [h,ax] = plotmatrix(X);
% s = size(h,1);
% for i = 1:s                                     % label the plots
%   xlabel(ax(s,i), i)
%   ylabel(ax(i,1), i)
% end

%X = [X (1:size(X,1))'];

distances = {'euclidean', 'cityblock', 'chebychev', 'correlation', 'spearman'};
linkages = {'single', 'average', 'complete'};
%distances = {'euclidean'};
%linkages = { 'average'};
% distance = 'euclidean';
% distance = 'cityblock';
% distance = 'chebychev';
distance = 'cosine';
% linkage_method = 'single'; %Shortest 
linkage_method = 'average'; %Unweighted average dis
% linkage_method = 'centroid'; %euclid only
% linkage_method = 'complete'; %Furthest 
% linkage_method = 'ward'; %euclid only
% linkage_method = 'weighted';
for i=1:length(distances)
    for j=1:length(linkages)
        distance = distances{i};
        linkage_method = linkages{j};
        h = figure;
        Y = pdist(X, distance);
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
end
%%
T = cluster(Z,'maxclust',7);
%T = cluster(Z,'cutoff',0.55);
for i = 1:size(X,1)
labels{i,2} = T(i);
end