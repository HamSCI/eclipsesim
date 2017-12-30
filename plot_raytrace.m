function plot_raytrace(tx_lat, tx_lon, ray_bearing, start_height, height_inc, iono_pf_grid, ray_path_data, UT)
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
    
    global ELEVS
    global R12
    global RANGE_INC
    global NUM_RANGES
    
    figure(1);

    UT_str = [num2str(UT(3)) '/' num2str(UT(2)) '/' num2str(UT(1)) ...
              '  ' num2str(UT(4), '%2.2d') ':' num2str(UT(5), '%2.2d') ...
              'UT'];
    elev_str = [num2str(ELEVS(1)) ' deg'];
    R12_str = num2str(R12);
    lat_str = num2str(tx_lat);
    lon_str = num2str(tx_lon);
    bearing_str = num2str(ray_bearing);
    fig_str = [UT_str '   elevation = ' elev_str '   R12 = ' R12_str ...
               '   lat = ' lat_str ', lon = ' lon_str ', bearing = ' ...
               bearing_str];
    start_range = 0;
    end_range = 2000;
    end_range_idx = fix(end_range ./ RANGE_INC) + 1;
    start_ht = start_height;
    start_ht_idx = 1;
    end_ht = 550;
    end_ht_idx = fix((end_ht - start_ht) ./ height_inc) + 1;
    
    set(gcf, 'name', fig_str);
    
    iono_pf_subgrid = iono_pf_grid(start_ht_idx:end_ht_idx, ...
                                   1:end_range_idx);
    [axis_handle, ray_handle] = ...
        plot_ray_iono_slice(iono_pf_subgrid, start_range, end_range, ...
                            RANGE_INC, start_ht, end_ht, height_inc, ...
                            ray_path_data, 'color', [1, 1, 0.99], ...
                            'linewidth', 2);

    caxis([0, 7.25]);
    
    set(gcf, 'units', 'normal')
    pos = get(gcf, 'position');
    pos(2) = 0.55;
    set(gcf, 'position', pos)

    % The following prints the figure to high-res encapsulated PS
    % and PNG files
    set(gcf, 'paperorientation', 'portrait')
    set(gcf, 'paperunits', 'cent', 'paperposition', [0 0 61 18])
    set(gcf, 'papertype', 'a4')
end