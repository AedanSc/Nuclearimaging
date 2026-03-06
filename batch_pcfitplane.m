%% batch_pcfitplane.m
% ─────────────────────────────────────────────────────────────────────────────
% Fits a plane to consecutive point cloud frames and tracks how the plane
% moves over time. Used for SPECT patient surface motion estimation.
%
% Lorenzo suggested using pcfitplane as a simpler starting algorithm before
% moving to more complex approaches.
%
% Output:
%   - Motion timeline printed to console
%   - Results saved to pcfitplane_results.csv
%   - Plots: displacement over time, normal vector drift
%
% Usage:
%   Run this script directly in MATLAB. Set the folder path below.
% ─────────────────────────────────────────────────────────────────────────────

clear; clc; close all;

%% ── Settings ────────────────────────────────────────────────────────────────
folder      = 'C:\Users\aedan\zividtopy\point_clouds\point_clouds\';
maxDistance = 10.0;  % mm — max distance from inlier point to plane (RANSAC)
voxelSize   = 5.0;   % mm — downsample voxel size (0 = no downsampling)
outputCSV   = 'pcfitplane_results.csv';

%% ── Load file list ───────────────────────────────────────────────────────────
files = dir(fullfile(folder, '*.ply'));
files = sort_nat({files.name});  % natural sort by filename = chronological

nFiles = numel(files);
if nFiles < 2
    error('Need at least 2 .ply files. Found %d.', nFiles);
end

fprintf('=============================================================\n');
fprintf('  Batch pcfitplane Motion Estimation\n');
fprintf('=============================================================\n');
fprintf('  Folder      : %s\n', folder);
fprintf('  Frames      : %d\n', nFiles);
fprintf('  Pairs       : %d\n', nFiles - 1);
fprintf('  Max distance: %.1f mm\n', maxDistance);
fprintf('  Voxel size  : %.1f mm\n', voxelSize);
fprintf('=============================================================\n\n');

%% ── Preallocate results ─────────────────────────────────────────────────────
nPairs          = nFiles - 1;
displacement_mm = zeros(nPairs, 1);
normal_drift    = zeros(nPairs, 1);
d_values        = zeros(nPairs, 2);   % plane D parameter (offset) per pair
normals         = zeros(nPairs, 3, 2); % normal vectors per pair
dt_ms           = zeros(nPairs, 1);
frame1_names    = cell(nPairs, 1);
frame2_names    = cell(nPairs, 1);
mean_errors     = zeros(nPairs, 2);

%% ── Process pairs ────────────────────────────────────────────────────────────
for i = 1:nPairs
    f1 = fullfile(folder, files{i});
    f2 = fullfile(folder, files{i+1});

    frame1_names{i} = files{i};
    frame2_names{i} = files{i+1};

    % Parse timestamps from filenames for time delta
    try
        t1 = datetime(files{i},   'InputFormat', '''point_cloud_''yyyyMMdd_HHmmss_SSSSSS''.ply''');
        t2 = datetime(files{i+1}, 'InputFormat', '''point_cloud_''yyyyMMdd_HHmmss_SSSSSS''.ply''');
        dt_ms(i) = milliseconds(t2 - t1);
    catch
        dt_ms(i) = NaN;
    end

    % Load point clouds
    pc1 = pcread(f1);
    pc2 = pcread(f2);

    % Downsample if requested
    if voxelSize > 0
        pc1 = pcdownsample(pc1, 'gridAverage', voxelSize);
        pc2 = pcdownsample(pc2, 'gridAverage', voxelSize);
    end

    % Denoise
    pc1 = pcdenoise(pc1);
    pc2 = pcdenoise(pc2);

    % Fit plane to both frames
[model1, ~, ~, err1] = pcfitplane(pc1, maxDistance, [0 0 1], 5, 'MaxNumTrials', 2000);
[model2, ~, ~, err2] = pcfitplane(pc2, maxDistance, [0 0 1], 5, 'MaxNumTrials', 2000);

    % Store plane parameters
    % Plane equation: ax + by + cz + d = 0
    % model.Parameters = [a b c d]
    params1 = model1.Parameters;
    params2 = model2.Parameters;

    n1 = params1(1:3);  % normal vector frame 1
    n2 = params2(1:3);  % normal vector frame 2
    d1 = params1(4);    % plane offset frame 1
    d2 = params2(4);    % plane offset frame 2

    % Fix normal sign ambiguity — force both normals to point in same direction
    % If dot product is negative, the normal flipped — correct it
    if dot(n1, n2) < 0
        n2 = -n2;
        d2 = -d2;
    end

    % Displacement = change in plane offset (distance between planes in mm)
    % This measures how far the surface moved along its normal direction
    disp_mm = abs(d2 - d1);
    displacement_mm(i) = disp_mm;

    % Normal vector drift = angle between normals (degrees)
    cos_angle = dot(n1, n2) / (norm(n1) * norm(n2));
    cos_angle = max(-1, min(1, cos_angle));  % clamp for numerical safety
    normal_drift(i) = rad2deg(acos(cos_angle));

    d_values(i, :)    = [d1, d2];
    normals(i, :, 1)  = n1;
    normals(i, :, 2)  = n2;
    mean_errors(i, :) = [err1, err2];

    % Print progress
    if ~isnan(dt_ms(i))
        fprintf('  [%04d/%04d] %s\n           Δt=%.1fms  disp=%.3fmm  normal_drift=%.3f°  err1=%.3f  err2=%.3f\n', ...
            i, nPairs, files{i}, dt_ms(i), disp_mm, normal_drift(i), err1, err2);
    else
        fprintf('  [%04d/%04d] %s\n           disp=%.3fmm  normal_drift=%.3f°\n', ...
            i, nPairs, files{i}, disp_mm, normal_drift(i));
    end
end

%% ── Summary ──────────────────────────────────────────────────────────────────
fprintf('\n═════════════════════════════════════════════════════════════\n');
fprintf('  Motion Summary\n');
fprintf('═════════════════════════════════════════════════════════════\n');
fprintf('  Mean displacement  : %.3f mm\n', mean(displacement_mm));
fprintf('  Max displacement   : %.3f mm\n', max(displacement_mm));
fprintf('  Min displacement   : %.3f mm\n', min(displacement_mm));
fprintf('  Std deviation      : %.3f mm\n', std(displacement_mm));
fprintf('  Mean normal drift  : %.3f°\n',   mean(normal_drift));
fprintf('═════════════════════════════════════════════════════════════\n');

%% ── Save CSV ─────────────────────────────────────────────────────────────────
T = table((1:nPairs)', frame1_names, frame2_names, dt_ms, ...
    displacement_mm, normal_drift, mean_errors(:,1), mean_errors(:,2), ...
    'VariableNames', {'pair','frame1','frame2','dt_ms', ...
    'displacement_mm','normal_drift_deg','plane_error_1','plane_error_2'});

writetable(T, outputCSV);
fprintf('\n  Results saved to: %s\n', fullfile(pwd, outputCSV));

%% ── Plots ────────────────────────────────────────────────────────────────────

% Build time axis in seconds from start
t_sec = cumsum([0; dt_ms(1:end-1)]) / 1000;

% Plot 1: Displacement over time
figure('Name', 'Surface Displacement Over Time', 'NumberTitle', 'off');
plot(t_sec, displacement_mm, 'b-o', 'MarkerSize', 3, 'LineWidth', 1.2);
xlabel('Time (s)');
ylabel('Displacement (mm)');
title('Patient Surface Displacement Over Time (pcfitplane)');
grid on;
yline(mean(displacement_mm), 'r--', sprintf('Mean: %.2f mm', mean(displacement_mm)), ...
    'LabelVerticalAlignment', 'bottom');

% Plot 2: Normal vector drift over time
figure('Name', 'Normal Vector Drift Over Time', 'NumberTitle', 'off');
plot(t_sec, normal_drift, 'r-o', 'MarkerSize', 3, 'LineWidth', 1.2);
xlabel('Time (s)');
ylabel('Normal Drift (degrees)');
title('Plane Normal Vector Drift Over Time (rotation proxy)');
grid on;
yline(mean(normal_drift), 'b--', sprintf('Mean: %.2f°', mean(normal_drift)), ...
    'LabelVerticalAlignment', 'bottom');

% Plot 3: Plane D parameter over time (raw offset tracking)
figure('Name', 'Plane Offset (D) Over Time', 'NumberTitle', 'off');
d_all = [d_values(:,1); d_values(end,2)];
plot(t_sec, d_values(:,1), 'k-', 'LineWidth', 1.2);
xlabel('Time (s)');
ylabel('Plane offset D (mm)');
title('Raw Plane Offset Over Time');
grid on;

fprintf('\n  Done! Check the figures and %s\n', outputCSV);

%% ── Helper: natural sort ─────────────────────────────────────────────────────
function sorted = sort_nat(c)
% Simple natural sort for cell array of strings
    [~, idx] = sort(c);
    sorted = c(idx);
end