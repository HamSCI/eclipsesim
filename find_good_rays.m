function [srch_ray_good, srch_frequency, srch_elevation, srch_group_range, ...
          srch_deviative_absorption, srch_D_Oabsorp, srch_D_Xabsorp, ...
          srch_fs_loss, srch_effective_range, srch_phase_path, ...
          srch_ray_apogee, srch_ray_apogee_gndr, srch_plasfrq_at_apogee, ...
          srch_ray_hops, srch_del_freq_O, srch_del_freq_X, srch_ray_data, ...
          srch_ray_path_data] = find_good_rays(srch_labels, srch_gnd_range, ...
                                               srch_grp_range, rx_range, freq, ...
                                               tx_lat, tx_lon, ray_bearing, UT)
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
%%----------------------------

    global NUM_HOPS
    global ELEVS
    global ELEV_STEP
    global TOL
    global CALC_IRREGS
    global R12

    srch_ray_good = 0;
    
    srch_frequency            = [];
    srch_elevation            = [];
    srch_group_range          = [];
    srch_deviative_absorption = [];
    srch_D_Oabsorp            = [];
    srch_D_Xabsorp            = [];
    srch_fs_loss              = [];
    srch_effective_range      = [];
    srch_phase_path           = [];
    srch_ray_apogee           = [];
    srch_ray_apogee_gndr      = [];
    srch_plasfrq_at_apogee    = [];
    srch_ray_hops             = [];
    srch_del_freq_O           = [];
    srch_del_freq_X           = [];
    srch_ray_data             = {};
    srch_ray_path_data        = {};
    
    for hop_idx = 1:NUM_HOPS        
        goodray_idx = find(srch_labels(:, hop_idx) == 1);
        
        if length(goodray_idx) > 3
            % Find ray ground ranges which bracket the RX ground
            % range, do raytracing with finer (0.05 deg) elevation
            % grid within coarse bracketing rays. Finally
            % interpolate to find the ray elevations and group rays
            % (and absorption losses) which will hit RX.
            srch_els       = ELEVS(goodray_idx);
            srch_gnd       = srch_gnd_range(goodray_idx, hop_idx)';
            srch_grp       = srch_grp_range(goodray_idx, hop_idx)';
            srch_dgrp_dels = deriv(srch_grp, srch_els);
            
            srch_grp_to_rx = [];

            for idx = 1:length(srch_els)-1
                % Find bracketing rays and ignore ones whose rate
                % of change of range with elevation is too large as
                % this indicates we are too far into a cusp region
                % to be reliable.
                if ((srch_gnd(idx) >= rx_range & srch_gnd(idx+1) < rx_range) | ...
                    (srch_gnd(idx) <= rx_range & srch_gnd(idx+1) > rx_range)) & ...
                        (srch_els(idx+1) - srch_els(idx) < 2*ELEV_STEP) & ...
                        (abs(srch_dgrp_dels(idx)) < 500) & (abs(srch_dgrp_dels(idx+1)) < 500)
                    
                    srch_el_step = srch_els(idx + 1) - srch_els(idx);
                    fine_el_step = srch_el_step ./ 5;
                    fine_els     = [srch_els(idx):fine_el_step:srch_els(idx + 1)];
                    fine_elevs   = [];
                    fine_gnd     = [];
                    fine_label   = [];

                    freqs = freq .* ones(size(fine_els));

                    % Raytrace at fine elevation steps between bracketing rays
                    [fine_ray_data, fine_ray_path_data] = ...
                        raytrace_2d(tx_lat, tx_lon, fine_els, ...
                                    ray_bearing, freqs, hop_idx, ...
                                    TOL, CALC_IRREGS);

                    for idx2 = 1:6
                        if fine_ray_data(idx2).nhops_attempted == hop_idx
                            fine_gnd = [fine_gnd fine_ray_data(idx2).ground_range(hop_idx)];
                            fine_label = [fine_label fine_ray_data(idx2).ray_label(hop_idx)];
                            fine_elevs = [fine_elevs fine_els(idx2)];
                        end
                    end

                    % Interpolate to get elevation to launch ray to hit RX and
                    % raytrace at this elevation to get all the other required
                    % values.
                    if (isempty(find(fine_label < 1)) & length(fine_label >= 3))
                        srch_elev_torx = interp1(fine_gnd, fine_elevs, rx_range, 'pchip');

                        % Do a raytrace 2D here...
                        [srch_ray_data, srch_ray_path_data] = ...
                            raytrace_2d(tx_lat, tx_lon, srch_elev_torx, ...
                                        ray_bearing, freq, hop_idx, TOL, ...
                                        CALC_IRREGS);

                        if srch_ray_data.ray_label == 1
                            srch_elevation = [srch_elevation srch_elev_torx];
                            srch_group_range = [srch_group_range ...
                                                srch_ray_data.group_range(hop_idx)];
                            srch_phase_path = [srch_phase_path ...
                                               srch_ray_data.phase_path(hop_idx)];
                            srch_deviative_absorption = [srch_deviative_absorption ...
                                                srch_ray_data.deviative_absorption(hop_idx)];
                            srch_effective_range = [srch_effective_range ...
                                                srch_ray_data.effective_range(hop_idx)];
                            
                            srch_gnd_fs_loss = 0;
                            srch_O_absorp    = 0;
                            srch_X_absorp    = 0;

                            for idx2 = 1:hop_idx
                                srch_ray_apogee = srch_ray_data.apogee(idx2);
                                srch_ray_apogee_gndr = srch_ray_data.gnd_rng_to_apogee(idx2);

                                [srch_ray_apogee_lat, srch_ray_apogee_lon] = ...
                                    raz2latlon(srch_ray_apogee_gndr, ...
                                               ray_bearing, tx_lat, tx_lon, ...
                                               'wgs84');

                                srch_plasfrq_at_apogee = srch_ray_data.plasma_freq_at_apogee(idx2);

                                if idx2 == 1
                                    % Calculate geomagnetic splitting factor,
                                    % assuming that it is the same for all hops
                                    % (really need to calculate separately for
                                    % each hop).
                                    [srch_del_fo, srch_del_fx] = ...
                                        gm_freq_offset(srch_ray_apogee_lat, srch_ray_apogee_lon, ...
                                                       srch_ray_apogee, ray_bearing, freq, ...
                                                       srch_plasfrq_at_apogee, UT);
                                end

                                srch_O_absorp = srch_O_absorp + abso_bg(srch_ray_apogee_lat, ...
                                                                        srch_ray_apogee_lon, ...
                                                                        srch_elev_torx, ...
                                                                        freq + srch_del_fo, ...
                                                                        UT, R12, 1);
                                srch_X_absorp = srch_X_absorp + abso_bg(srch_ray_apogee_lat, ...
                                                                        srch_ray_apogee_lon, ...
                                                                        srch_elev_torx, ...
                                                                        freq + srch_del_fo, ...
                                                                        UT, R12, 0);

                                if idx2 > 1
                                    srch_fs_lat = srch_ray_data.lat(idx2 - 1);
                                    srch_fs_lon = srch_ray_data.lon(idx2 - 1);
                                    srch_gnd_fs_loss = srch_gnd_fs_loss + ...
                                        ground_fs_loss(srch_fs_lat, srch_fs_lon, ...
                                                       srch_elev_torx, freq);
                                end
                            end

                            srch_D_Oabsorp = [srch_D_Oabsorp srch_O_absorp];
                            srch_D_Xabsorp = [srch_D_Xabsorp srch_X_absorp];
                            srch_fs_loss = [srch_fs_loss srch_gnd_fs_loss];
                            srch_del_freq_O = [srch_del_freq_O srch_del_fo];
                            srch_del_freq_X = [srch_del_freq_X srch_del_fx];
                            srch_frequency = [srch_frequency freq];
                            srch_ray_hops = [srch_ray_hops hop_idx];
                            srch_ray_good = 1;
                        end % srch_ray_data.ray_label == 1
                    end % (isempty(find(fine_label < 1))...
                end % ((gnd_rng(idx) >= rx_range...
            end % idx = 1:length(elevs)-1
        end % length(goodray_idx) > 3
    end % hop_idx = 1:NUM_HOPS
end
