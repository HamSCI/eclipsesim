function eclipse(job_name, job_id, plot, eclipse, out_path, plot_path, sami3_path)
%%----------------------------
%%  Copyright (C) 2017 The Center for Solar-Terrestrial Research at
%%                     the New Jersey Institute of Technology
%%
%%  This program is free software: you can redistribute it and/or modify
%%  it under the terms of the GNU General Public License as published by
%%  the Free Software Foundation, either version 3 of the License, or
%%  (at your option) any later version.
%%
%%  This program is distributed in the hope that it will be useful,
%%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%%  GNU General Public License for more details.
%%
%%  You should have received a copy of the GNU General Public License
%%  along with this program.  If not, see <http://www.gnu.org/licenses/>.
%%
%%
%%
%% REFACTOR NOTES:
%%  - geomagnetic splitting factor is calculated twice.
%%  
%% TODO:
%%  - Check for previous progress and resume where left off (compare job files
%%    and output files).
%%----------------------------

    % Global constants (that aren't actually global apparently cause MATLAB)
    global SPEED_OF_LIGHT
    global ELEV_STEP
    global NUM_HOPS
    global ELEVS
    global TX_POWER
    global GAIN_TX_DB
    global GAIN_RX_DB
    global R12
    global CALC_DOPPLER
    global CALC_IRREGS
    global KP
    global MAX_RANGE
    global NUM_RANGES
    global TOL
    global RANGE_INC
    global START_HEIGHT
    global HEIGHT_INC
    global NUM_HEIGHTS
    global BANDS
    
    SPEED_OF_LIGHT = 2.99792458e8;          % The speed of light (m/s)
    ELEV_STEP = 0.5;                        % TBD
    NUM_HOPS = 3;                           % TBD
    ELEVS = [5:ELEV_STEP:60];               % TBD
    TX_POWER = 1;                           % TBD
    GAIN_TX_DB = 1;                         % TBD
    GAIN_RX_DB = 1;                         % TBD 
    R12 = -1;                               % TBD
    CALC_DOPPLER = 0;                       % Generate ionosphere 5
                                            % minutes later so that the
                                            % doppler shift can be
                                            % calculated (1 = true; 0 =
                                            % false)
    CALC_IRREGS = 0;                        % TBD
    KP = 0;                                 % TBD
    MAX_RANGE = 10000;                      % Maximum range (km) to
                                            % sample the ionosphere
    NUM_RANGES = 201;                       % The number of ranges to
                                            % calculate (must be <
                                            % 2000)
    
    TOL = 1e-7;                             % ODE tolerance
    RANGE_INC = MAX_RANGE ./ (NUM_RANGES - 1); % Range cell size (km)
    START_HEIGHT = 0;
    HEIGHT_INC = 3;
    NUM_HEIGHTS = 200;
    BANDS = [
        1.830
        3.530
        7.030
        14.030
        21.030
        28.030
            ]                           % Bands to test

    JOB_PATH   = './jobs/';
    JOB_ID     = job_id;
    OUT_PATH   = out_path;
    PLT_PATH   = plot_path;
    SAMI3_PATH = sami3_path;

    file_name = job_name(8:end)
    
    timestamp  = file_name(1:16);
    year_str   = file_name(1:4);
    month_str  = file_name(6:7);
    day_str    = file_name(9:10);
    hour_str   = file_name(12:13);
    minute_str = file_name(15:16);
    
    UT = [str2num(year_str) str2num(month_str) str2num(day_str) ...
          str2num(hour_str) str2num(minute_str)];
    
    disp('Loading ionosphere...');
    
    load(strcat(SAMI3_PATH, 'grid.mat'));
    load(strcat(SAMI3_PATH, 'data_', num2str(JOB_ID, '%04u'), '.mat'));
    
    if eclipse ~= 0
        interpolator = scatteredInterpolant(double(grid_lats(:)), ...
                                            double(grid_lons(:)), ...
                                            double(grid_heights(:)), ...
                                            double(eclipse_data(:)), ...
                                            'natural');
    else
        interpolator = scatteredInterpolant(double(grid_lats(:)), ...
                                            double(grid_lons(:)), ...
                                            double(grid_heights(:)), ...
                                            double(base_data(:)), ...
                                            'natural');
    end

    disp('Done.');

    irreg = zeros(4, NUM_RANGES);

    csv_file = fopen(strcat(JOB_PATH, file_name), 'r');

    if eclipse ~= 0
        out_file = fopen(strcat(OUT_PATH, 'eclipse/simulated_', file_name), 'w');
    else
        out_file = fopen(strcat(OUT_PATH, 'base/simulated_', file_name), 'w');
    end
    
    % Write header
    fprintf(out_file, ['tx_call,' ...
                       'rx_call,' ...
                       'tx_lat,' ...
                       'tx_lon,' ...
                       'rx_lat,' ...
                       'rx_lon,' ...
                       'freq,' ...
                       'srch_rd_lat,' ...
                       'srch_rd_lon,' ...
                       'srch_rd_ground_range,' ...
                       'srch_rd_group_range,' ...
                       'srch_rd_phase_path,' ...
                       'srch_rd_geometric_path_length,' ...
                       'srch_rd_initial_elev,' ...
                       'srch_rd_final_elev,' ...
                       'srch_rd_apogee,' ...
                       'srch_rd_gnd_rng_to_apogee,' ...
                       'srch_rd_plasma_freq_at_apogee,' ...
                       'srch_rd_virtual_height,' ...
                       'srch_rd_effective_range,' ...
                       'srch_rd_deviative_absorption,' ...
                       'srch_rd_TEC_path,' ...
                       'srch_rd_Doppler_shift,' ...
                       'srch_rd_Doppler_spread,' ...
                       'srch_rd_FAI_backscatter_loss,' ...
                       'srch_rd_frequency,' ...
                       'srch_rd_nhops_attempted,' ...
                       'srch_rd_rx_power_0_dB,' ...
                       'srch_rd_rx_power_dB,' ...
                       'srch_rd_rx_power_O_dB,'...
                       'srch_rd_rx_power_X_dB,'...
                       'srch_rd_hop_idx,' ...
                       'srch_rd_apogee_lat,' ...
                       'srch_rd_apogee_lon' ...
                       '\n']);

    % Our input files have headers
    fgets(csv_file);

    while ~feof(csv_file)
        line = fgets(csv_file);         % Read the input file
                                        % line-by-line

        data = strsplit(line, ',');

        tx_call = cell2mat(data(1))
        rx_call = cell2mat(data(2))

        tx_lat = str2num(cell2mat(data(3)));
        tx_lon = str2num(cell2mat(data(4)));
        rx_lat = str2num(cell2mat(data(5)));
        rx_lon = str2num(cell2mat(data(6)));

        [rx_range, rx_azm] = latlon2raz(rx_lat, rx_lon, tx_lat, tx_lon);
        rx_range = rx_range / 1000.;    % Convert range to km
        ray_bearing = rx_azm;           % Assume omni-directional antenna
                                        % (no coning)

        slice_params = [tx_lat tx_lon rx_lat rx_lon NUM_RANGES RANGE_INC START_HEIGHT HEIGHT_INC NUM_HEIGHTS];

        iono_en_grid_2d = create_2d_slice(interpolator, slice_params);
        iono_pf_grid_2d = real((iono_en_grid_2d * 80.6164e-6) .^ 0.5);

        slice_size = size(iono_en_grid_2d);

        collision_freq_2d = zeros(slice_size(1), NUM_RANGES);

        for band_idx = 1:length(BANDS)
            freq = BANDS(band_idx)
            freqs = freq .* ones(size(ELEVS));
            
            [ray_data, ray_path_data] = ...
                raytrace_2d(tx_lat, rx_lon, ELEVS, ray_bearing, freqs, ...
                            NUM_HOPS, TOL, CALC_IRREGS, iono_en_grid_2d, ...
                            iono_en_grid_2d, collision_freq_2d, START_HEIGHT, ...
                            HEIGHT_INC, RANGE_INC, irreg);
            
            if plot ~= 0
                plottable = 1;
                
                for ray_idx = 1:length(ray_path_data)
                    if isempty(ray_path_data(ray_idx).ground_range)
                        continue
                    end
                    
                    if isempty(ray_path_data(ray_idx).height)
                        continue
                    end
                    
                    if length(ray_path_data(ray_idx).ground_range) ~= ...
                            length(unique(ray_path_data(ray_idx).ground_range))
                        plottable = 0
                    end
                end
                
                %%
                %% Plot Raytrace
                %%
                if plottable ~= 0
                    plot_raytrace(tx_lat, tx_lon, ray_bearing, START_HEIGHT, HEIGHT_INC, ...
                                  iono_pf_grid_2d, ray_path_data, UT);
                    
                    print('-dpng', strcat(PLT_PATH, timestamp, '-', ...
                                          tx_call, '-', rx_call, '_', ...
                                          num2str(freq), '.png'));
                end
            end
            
            %%
            %% Identify Ray Hitting Receiver
            %%
            num_elevs = length(ELEVS);
            
            srch_gnd_range = zeros(num_elevs, NUM_HOPS) * NaN;
            srch_grp_range = zeros(num_elevs, NUM_HOPS) * NaN;
            srch_labels    = zeros(num_elevs, NUM_HOPS);
            
            % Loop over ray elevation with 0.5 deg steps
            for elev_idx = 1:num_elevs
                for hop_idx = 1:ray_data(elev_idx).nhops_attempted
                    srch_gnd_range(elev_idx, hop_idx) = ...
                        ray_data(elev_idx).ground_range(hop_idx);
                    srch_grp_range(elev_idx, hop_idx) = ...
                        ray_data(elev_idx).group_range(hop_idx);
                    srch_labels(elev_idx, hop_idx) = ...
                        ray_data(elev_idx).ray_label(hop_idx);
                end
            end

            srch_ray_good = 0;

            % Find the rays that have come to ground
            [srch_ray_good, srch_frequency, srch_elevation, srch_group_range, ...
             srch_deviative_absorption, srch_D_Oabsorp, srch_D_Xabsorp, ...
             srch_fs_loss, srch_effective_range, srch_phase_path, ...
             srch_ray_apogee, srch_ray_apogee_gndr, srch_plasfrq_at_apogee, ...
             srch_ray_hops, srch_del_freq_O, srch_del_freq_X, srch_ray_data, ...
             srch_ray_path_data] = ...
                find_good_rays(srch_labels, srch_gnd_range, ...
                               srch_grp_range, rx_range, freq, ...
                               tx_lat, tx_lon, ray_bearing, UT);

            if srch_ray_good ~= 0
                % Determine number of separate ray segments
                srch_rd_points = 0;
                srch_num_elevs = length(srch_ray_data);
                
                for idx = 1:srch_num_elevs
                    srch_rd_points = srch_rd_points + length(srch_ray_data(idx).lat);
                end
                
                % Create index vector of ray segments
                srch_rd_id = zeros(1, srch_rd_points);
                start_idx = 1;
                
                for idx = 1:srch_num_elevs
                    n = length(srch_ray_data(idx).lat);
                    end_idx = start_idx + n - 1;
                    srch_rd_id(start_idx:end_idx) = idx;
                    start_idx = end_idx + 1;
                end
                
                % Declare new non-structure vectors for each field name
                srch_ray_data_fieldnames = fieldnames(srch_ray_data);
                fns = srch_ray_data_fieldnames;
                
                for idx = 1:length(fns)
                    expr = strcat('srch_rd_', fns(idx), ...
                                  '= zeros(1,', num2str(srch_rd_points),');');
                    eval(expr{1});
                end
                
                % Populate the newly declared vectors
                start_idx = 1;
                
                for idx = 1:srch_num_elevs
                    n = length(srch_ray_data(idx).lat);
                    end_idx = start_idx + n - 1;
                    
                    for idx2 = 1:length(fns)
                        % We are not currently interested in FAIs...
                        if ~strcmp(fns(idx2), 'FAI_backscatter_loss')
                            lhs = strcat('srch_rd_', fns(idx2), ...
                                         '(', num2str(start_idx), ':', ...
                                         num2str(end_idx), ')');
                            rhs = strcat('srch_ray_data(', num2str(idx), ').', ...
                                         fns(idx2));
                            expr = strcat(lhs{1}, '=', rhs{1}, ';');
                            eval(expr);
                        end
                    end
                    
                    start_idx = end_idx + 1;
                end
                
                % Power calculations
                srch_rd_D_Oabsorp = zeros(1, length(srch_rd_effective_range)) * NaN;
                srch_rd_D_Xabsorp = zeros(1, length(srch_rd_effective_range)) * NaN;
                srch_rd_fs_loss   = zeros(1, length(srch_rd_effective_range)) * NaN;
                
                id = 0;
                for idx = 1:length(srch_ray_data)
                    gnd_fs_loss = 0;
                    O_absorp    = 0;
                    X_absorp    = 0;
                    
                    for idx2 = 1:length(srch_ray_data(idx).ray_label)
                        id = id + 1;
                        
                        if srch_ray_data(idx).ray_label(idx2) < 1
                            continue;
                        end
                        
                        ray_apogee = srch_rd_apogee(id);
                        ray_apogee_gndr = srch_rd_gnd_rng_to_apogee(id);
                        
                        [ray_apogee_lat, ray_apogee_lon] = ...
                            raz2latlon(ray_apogee_gndr, ray_bearing, ...
                                       tx_lat, tx_lon, 'wgs84');
                        
                        plasfrq_at_apogee = srch_rd_plasma_freq_at_apogee(id);
                        
                        if idx2 == 1
                            % Calculate geomagnetic splitting factor and assume
                            % that it is the same for all hops (really need to
                            % calculate separately for each hop).
                            [del_fo del_fx] = ...
                                gm_freq_offset(ray_apogee_lat, ray_apogee_lon, ...
                                               ray_apogee, ray_bearing, freq, ...
                                               plasfrq_at_apogee, UT);
                        end
                        
                        elev = srch_rd_initial_elev(id);
                        O_absorp = O_absorp + abso_bg(ray_apogee_lat, ray_apogee_lon, ...
                                                      elev, freq + del_fo, UT, R12, 1);
                        X_absorp = X_absorp + abso_bg(ray_apogee_lat, ray_apogee_lon, ...
                                                      elev, freq + del_fx, UT, R12, 0);
                        
                        srch_rd_D_Oabsorp(id) = O_absorp;
                        srch_rd_D_Xabsorp(id) = X_absorp;
                        
                        if idx2 > 1
                            fs_lat = srch_rd_lat(id - 1);
                            fs_lon = srch_rd_lon(id - 1);
                            
                            % Forward ground-scattering loss
                            gnd_fs_loss = gnd_fs_loss + ground_fs_loss(fs_lat, fs_lon, ...
                                                                       elev, freq);
                        end
                        
                        srch_rd_fs_loss(id) = gnd_fs_loss;
                    end
                end
                
                % One-way RADAR equation
                wavelen = SPEED_OF_LIGHT ./ (freq .* 1e6);
                pwr_tmp = TX_POWER * (wavelen .^ 2 ./ (4 .* pi)) ./ ...
                          (4 .* pi .* srch_rd_effective_range .^ 2);
                
                srch_rd_rx_power_0_dB = 10 * log10(pwr_tmp) + GAIN_TX_DB + GAIN_RX_DB;
                srch_rd_rx_power_dB = srch_rd_rx_power_0_dB - srch_rd_deviative_absorption - ...
                    srch_rd_fs_loss;
                srch_rd_rx_power_O_dB = srch_rd_rx_power_dB - srch_rd_D_Oabsorp;
                srch_rd_rx_power_X_dB = srch_rd_rx_power_dB - srch_rd_D_Xabsorp;
            
                % Put the data into a CSV file to be processed later.
                for i = 1:length(srch_rd_lat)
                    % Calculate apogee coordinates
                    [rng, azm] = latlon2raz(srch_rd_lat(i), srch_rd_lon(i), tx_lat, tx_lon);
                    
                    [apogee_lat, apogee_lon] = raz2latlon(srch_rd_ground_range(i) * 1000, ...
                                                          azm, tx_lat, tx_lon);
                    
                    fprintf(out_file, '%s,%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n', ...
                            tx_call, ...
                            rx_call, ...
                            tx_lat, ...
                            tx_lon, ...
                            rx_lat, ...
                            rx_lon, ...
                            freq, ...
                            srch_rd_lat(i), ...
                            srch_rd_lon(i), ...
                            srch_rd_ground_range(i), ...
                            srch_rd_group_range(i), ...
                            srch_rd_phase_path(i), ...
                            srch_rd_geometric_path_length(i), ...
                            srch_rd_initial_elev(i), ...
                            srch_rd_final_elev(i), ...
                            srch_rd_apogee(i), ...
                            srch_rd_gnd_rng_to_apogee(i), ...
                            srch_rd_plasma_freq_at_apogee(i), ...
                            srch_rd_virtual_height(i), ...
                            srch_rd_effective_range(i), ...
                            srch_rd_deviative_absorption(i), ...
                            srch_rd_TEC_path(i), ...
                            srch_rd_Doppler_shift(i), ...
                            srch_rd_Doppler_spread(i), ...
                            srch_rd_FAI_backscatter_loss(i), ...
                            srch_rd_frequency(i), ...
                            srch_rd_nhops_attempted(i), ...
                            srch_rd_rx_power_0_dB(i), ...
                            srch_rd_rx_power_dB(i), ...
                            srch_rd_rx_power_O_dB(i), ...
                            srch_rd_rx_power_X_dB(i), ...
                            i, ...
                            apogee_lat, ...
                            apogee_lon);
                end
            end
        end
    end
    
    fclose(out_file);
    fclose(csv_file);
end
