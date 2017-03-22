a = [8 4; 12 15; 16 36; 21 50];
x = a(:,1)';
y = a(:,2)';
p = polyfit(x, y, 2);
i = 7:40;
v = polyval(p, i);
figure;hold all;
plot(i, v);
a = [8 10; 12 40; 16 90; 21 149];
x2 = a(:,1)';
y2 = a(:,2)';
p = polyfit(x2, y2, 2);
v = polyval(p, i);
plot(i, v);

a = [8 30; 12 60; 16 142; 21 235];
x3 = a(:,1)';
y3 = a(:,2)';
p = polyfit(x3, y3, 2);
v = polyval(p, i);
plot(i, v);

plot(x,y, 'o');
plot(x2,y2, 'o');
plot(x3,y3, 'o');
xlabel('Maze width/height')
ylabel('Convergence epoch')
legend('no wall', '1 wall', '2 walls');