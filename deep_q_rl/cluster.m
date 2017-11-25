labels = {'Alien'
'Assault'
'Asterix'
'Atlantis'
'Battle Zone'
'Bowling'
'Boxing'
'Breakout'
'Crazy Climber'
'Demon Attack'
'Enduro'
'Fishing Derby'
'Frostbite'
'Gopher'
'Ice Hockey'
'Kangaroo'
'Krull'
'Kung-Fu Master'
'Name This Game'
'Pong'
'Private Eye'
'Q*bert'
'River Raid'
'Seaquest'
'Space Invaders'
'Star Gunner'
'Tennis'
'Time Pilot'
'Venture'
'Video pinball'
'Wizard of Wor'
'Zaxxon'
};

[h,ax] = plotmatrix(X);
s = size(h,1);
for i = 1:s                                     % label the plots
  xlabel(ax(s,i), i)
  ylabel(ax(i,1), i)
end

figure
Y = pdist(X, 'hamming');
Z = linkage(Y);
dendrogram(Z, 'labels', labels)

set(gca,'XTickLabelRotation',45)
