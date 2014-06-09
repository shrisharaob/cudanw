clear vm cm v i;
dt = 0.025;

vm = importdata('vm.csv');
v = vm(:, [1:2:end]);
i = vm(:, [2:2:end]);
plot(v);
cm = importdata('conMat.csv')
t = dt: dt: dt * size(v, 1);
for r = 1 : size(v, 2) 
    for c = 1 : size(v, 2)
        if((cm(r, c) == 1) && (r ~= c))
            plot(t, v(:, r), 'r', t, v(:, c), 'k');
            drawnow;
            title([num2str(r), '-->', num2str(c)]);
            waitforbuttonpress;
            clf;
        end
    end
end
