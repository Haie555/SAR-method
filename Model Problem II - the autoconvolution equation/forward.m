
function y = forward(x)
    N = length(x);
    h = 1/(N-1);
    y = zeros(N, 1);

    for j = 2:N
        z = 0;
        for i = 1:j-1
            % 使用梯形法来提高积分精度
            z = z + 0.5 * h * (x(j-i) * x(i) + x(j-i+1) * x(i+1));
        end
        y(j) = z;
    end
end
