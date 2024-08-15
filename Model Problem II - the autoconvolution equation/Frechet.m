
function A = Frechet(x)
    N = length(x);
    h = 1/(N-1);
    A = zeros(N, N);

    for i = 2:N
        A(i, 1) = h * x(i);
        A(i, i) = h * x(1);
        for j = 2:i-1
            A(i, j) = 2 * h * x(i-j+1);
        end
    end
end
