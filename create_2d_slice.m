function [iono_en_grid] = create_2d_slice(interpolator, params)
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

    tx_lat = params(1);
    tx_lon = params(2);
    rx_lat = params(3);
    rx_lon = params(4);

    num_ranges   = params(5);
    range_inc    = params(6);
    height_start = params(7);
    height_inc   = params(8);
    num_heights  = params(9);

    height_stop = height_start + ((num_heights - 1) * height_inc);

    lats = [];
    lons = [];
    heights = height_start:height_inc:height_stop;

    [rx_range, rx_azm] = latlon2raz(rx_lat, rx_lon, tx_lat, tx_lon);

    iono_en_grid = zeros(num_heights, num_ranges);

    for i = 1:num_ranges
        [lat, lon] = raz2latlon(range_inc * (i - 1) * 1000, rx_azm, tx_lat, tx_lon);

        for j = 1:num_heights
            iono_en_grid(j, i) = interpolator({lat, lon, (j - 1) * height_inc});
        end
    end
end