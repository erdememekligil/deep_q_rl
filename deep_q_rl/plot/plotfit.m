a = [8 10; 12 40; 16 90; 21 149];
x = a(:,1)';
y = a(:,2)';
p = polyfit(x, y, 2)
i = 7:40;
v = polyval(p, i);
figure;hold all;
plot(i, v);
plot(x,y, 'o');