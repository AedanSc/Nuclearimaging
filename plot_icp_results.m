%% plot_icp_results.m
% ─────────────────────────────────────────────────────────────────────────────
% Loads the ICP results CSV from batch_icp.py and generates clean plots
% for the motion timeline. Used for SPECT patient surface motion analysis.
%
% Usage:
%   Run this script directly in MATLAB after running batch_icp.py in Python.
% ─────────────────────────────────────────────────────────────────────────────

clear; clc; close all;

%% ── Load CSV ─────────────────────────────────────────────────────────────────
csvPath = 'C:\Users\aedan\zividtopy\icp_results.csv';

if ~isfile(csvPath)
    error('icp_results.csv not found. Run batch_icp.py first.');
end

T = readtable(csvPath);

% Remove the outlier pair (152 second gap between sessions)
gap_idx = T.dt_ms > 10000;
T_clean = T(~gap_idx, :);
T_gap   = T(gap_idx, :);

fprintf('Loaded %d pairs (%d removed as session gaps)\n', height(T), sum(gap_idx));

%% ── Build time axis ──────────────────────────────────────────────────────────
% Cumulative time in seconds from first frame
t_ms  = cumsum([0; T_clean.dt_ms(1:end-1)]);
t_sec = t_ms / 1000;

disp_mm = T_clean.displacement_mm;
rot_deg = T_clean.rotation_deg;
fitness = T_clean.fitness;
rmse    = T_clean.rmse;
tx      = T_clean.tx_mm;
ty      = T_clean.ty_mm;
tz      = T_clean.tz_mm;

%% ── Plot 1: Displacement over time ──────────────────────────────────────────
figure('Name', 'ICP Displacement Over Time', 'NumberTitle', 'off', ...
    'Position', [100 100 1000 400]);

plot(t_sec, disp_mm, 'b-', 'LineWidth', 1.0);
hold on;
yline(mean(disp_mm), 'r--', 'LineWidth', 1.5, ...
    'Label', sprintf('Mean: %.2f mm', mean(disp_mm)), ...
    'LabelVerticalAlignment', 'bottom');
xlabel('Time (s)', 'FontSize', 12);
ylabel('Displacement (mm)', 'FontSize', 12);
title('Patient Surface Displacement Over Time (ICP)', 'FontSize', 14);
grid on;
xlim([0 max(t_sec)]);
ylim([0 max(disp_mm) * 1.1]);

fprintf('\nDisplacement stats:\n');
fprintf('  Mean : %.3f mm\n', mean(disp_mm));
fprintf('  Max  : %.3f mm\n', max(disp_mm));
fprintf('  Min  : %.3f mm\n', min(disp_mm));
fprintf('  Std  : %.3f mm\n', std(disp_mm));

%% ── Plot 2: X Y Z translation components over time ──────────────────────────
figure('Name', 'ICP Translation Components', 'NumberTitle', 'off', ...
    'Position', [100 550 1000 400]);

plot(t_sec, tx, 'r-', 'LineWidth', 1.0, 'DisplayName', 'X');
hold on;
plot(t_sec, ty, 'g-', 'LineWidth', 1.0, 'DisplayName', 'Y');
plot(t_sec, tz, 'b-', 'LineWidth', 1.0, 'DisplayName', 'Z');
yline(0, 'k--', 'LineWidth', 0.8);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Translation (mm)', 'FontSize', 12);
title('ICP Translation Components Over Time (X, Y, Z)', 'FontSize', 14);
legend('Location', 'best');
grid on;
xlim([0 max(t_sec)]);

%% ── Plot 3: Rotation over time ───────────────────────────────────────────────
figure('Name', 'ICP Rotation Over Time', 'NumberTitle', 'off', ...
    'Position', [100 1000 1000 300]);

plot(t_sec, rot_deg, 'm-', 'LineWidth', 1.0);
hold on;
yline(mean(rot_deg), 'k--', 'LineWidth', 1.5, ...
    'Label', sprintf('Mean: %.3f°', mean(rot_deg)), ...
    'LabelVerticalAlignment', 'bottom');
xlabel('Time (s)', 'FontSize', 12);
ylabel('Rotation (degrees)', 'FontSize', 12);
title('ICP Rotation Over Time', 'FontSize', 14);
grid on;
xlim([0 max(t_sec)]);

%% ── Plot 4: ICP fitness and RMSE ─────────────────────────────────────────────
figure('Name', 'ICP Quality Metrics', 'NumberTitle', 'off', ...
    'Position', [1150 100 800 600]);

subplot(2,1,1);
plot(t_sec, fitness, 'g-', 'LineWidth', 1.0);
yline(mean(fitness), 'r--', 'Label', sprintf('Mean: %.3f', mean(fitness)));
xlabel('Time (s)');
ylabel('Fitness');
title('ICP Fitness Score (1.0 = perfect)');
grid on;
xlim([0 max(t_sec)]);
ylim([0 1.05]);

subplot(2,1,2);
plot(t_sec, rmse, 'r-', 'LineWidth', 1.0);
yline(mean(rmse), 'b--', 'Label', sprintf('Mean: %.3f mm', mean(rmse)));
xlabel('Time (s)');
ylabel('RMSE (mm)');
title('ICP Inlier RMSE');
grid on;
xlim([0 max(t_sec)]);

%% ── Plot 5: Cumulative displacement ──────────────────────────────────────────
figure('Name', 'Cumulative Displacement', 'NumberTitle', 'off', ...
    'Position', [1150 750 800 300]);

plot(t_sec, cumsum(disp_mm), 'k-', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Cumulative Displacement (mm)', 'FontSize', 12);
title('Cumulative Surface Displacement Over Time', 'FontSize', 14);
grid on;
xlim([0 max(t_sec)]);

fprintf('\nDone! %d plots generated.\n', 5);
