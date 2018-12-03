function [iono_en_grid] = create_2d_slice_tec(interpolator, params)
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
    
    global R12

    tx_lat = params(1);
    tx_lon = params(2);
    rx_lat = params(3);
    rx_lon = params(4);

    num_ranges   = params(5);
    range_inc    = params(6);
    height_start = params(7);
    height_inc   = params(8);
    num_heights  = params(9);
    
    UT      = params(10);
    use_tec = params(11);

    height_stop = height_start + ((num_heights - 1) * height_inc);

    lats    = [];
    lons    = [];
    heights = height_start:height_inc:height_stop;

    [rx_range, rx_azm] = latlon2raz(rx_lat, rx_lon, tx_lat, tx_lon);

    iono_en_grid = zeros(num_heights, num_ranges);

    for i = 1:num_ranges
        [lat, lon] = raz2latlon(range_inc * (i - 1) * 1000, rx_azm, tx_lat, ...
                                tx_lon);
        
        % Use the GPS-TEC data to adjust the IRI model. See Krankowski,
        % Shagimuratov, and Baran (2007) for k=1.61 value.
        iri_opts = {}
        
        if use_tec == 1
            tec_val = interpolator({lat, lon, UT});
            iri_opts.foF2 = 1.61 * (tec_val .^ 0.5);
        end
        
        % Get the IRI2016 result for the selected location and timestamp.
        [iri_data, iri_extra] = iri2016(lat, lon, R12, UT, height_start ...
                                        height_inc, num_heights, iri_opts);
        
        iono_en_grid(:, i) = iri_data(1, :);
    end
end