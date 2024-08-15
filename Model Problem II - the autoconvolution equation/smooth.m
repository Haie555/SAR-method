function y = smooth(x, window_size)
    y = x;
    half_window = floor(window_size / 2);
    for i = 1:length(x)
        start_idx = max(1, i - half_window);
        end_idx = min(length(x), i + half_window);
        y(i) = mean(x(start_idx:end_idx));
    end
end