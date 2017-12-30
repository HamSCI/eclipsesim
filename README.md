# Eclipse Simulation

This project is an ionospheric raytracer that outputs all successful hops into a
CSV file to be processed later. Additionally, this project supports the
simulation of the 2017 "Great American Eclipse" \[1\].

This project uses PHaRLAP by M. A. Cervera \[2\] for raytracing and
eclipse2017_ionopath by M. L. Moses \[3\] for simulating the eclipse.

## External Libraries

In order to use this project, you must place PHaRLAP into a subdirectory
`./pharlap_4.1.3` and eclipse2017_ionopath into `./eclipse2017_ionopath`.

## Getting Started

In order to run the eclipse simulation you must place your input CSV files
(called jobs) into the `./jobs` directory. Each job filename must be in the
format `YYYY-MM-DD hh:mm:ss.csv`. Each row of the job file must contain a TX
callsign, RX callsign, TX latitude and longitude, and RX latitude and longitude.

In order to run the project, you must execute one of the bash scripts:

- `eclipsesim_local.sh` is designed to run on a personal computer and will only
  compute the first job file. This script is ideal for single-job executions and
  testing.
- `eclipsesim_sge.sh` is designed to run on a distributed computing cluster
  using the Son of Grid Engine (SGE) job scheduler and is the recommended method
  for multi-job executions.

The SGE bash script contains additional functionality in order to allow the
project to run across a distributed computing cluster. Because of this, there
are a few additional settings that must be configured. The most important of
these is the `#$ -t` line. This options tells the job scheduler how many
sub-jobs must be started. This should be the number of job files you want to
process for maximum speed (`1-N` where `N` is the number of job files in
`./jobs`). For more information on executing the project on an SGE cluster,
please see the SGE documentation and the cluster owner's additional
documentation.

By default, both execution scripts will raytrace an "un-eclipsed"
ionosphere. This is to say that the eclipse2017_ionopath code will not be
executed, resulting in a "normal" ionosphere. In order to enable the eclipsed
ionosphere, open the respective execution script, find the `ECLIPSE=0` line, and
replace the `0` with `1`.

By default, individual plots of each raytrace are disabled. This is because the
generated plots can quickly accumulate, potentially resulting in several
gigabytes of images. To enable the plot generation, open the respective
execution script, find the `PLOTS=0` line, and replace the `0` with `1`.

By default, the output plots (if enabled) and CSV traces are outputted to
subdirectories within the current director. Thse can be changed by changing the
`PLOT_PATH` and `OUT_PATH` variables, respectively.

## How It Works

The project begins by loading in a job file line by line and separating the
different columns. A two-dimensional ionosphere is then generated between the TX
and RX coordinates through PHaRLAP using IRI 2016. Once this has been done, the
eclipsed attentuation function may be taken into account using
eclipse2017_ionopath (if `ECLIPSE=1` in the execution script) and each band is
taken and raytraced (and plotted if `PLOTS=1` in the execution script) through
the generated ionosphere using PHaRLAP.

For each band at each elevation, the raytraced rays that contact the ground are
re-raytraced with a finer level of elevation detail and then the "best" ray for
each hop is written into the output CSV file. Therefore, each row in the output
CSV file is a single ray hop.

For information on most of the output CSV file columns, see the PHaRLAP
documentation for `raytrace_2d`.

## License

Copyright (C) 2017 The Center for Solar-Terrestrial Research at
                   the New Jersey Institute of Technology

This project is licensed under the GNU GPL version 3. Please see `LICENSE.md`
for details.

## Credits

- Dr. Nathaniel A. Frissell (W2NAF) <nathaniel.a.frissell@njit.edu>
- Joshua D. Katz (KD2JAO) <jk369@njit.edu>
- Joshua S. Vega (KD2NKK) <jsv28@njit.edu>

## References

\[1\] https://www.greatamericaneclipse.com

\[2\] http://www.ursi.org/proceedings/procGA11/ursi/G07-7.pdf

\[3\] https://github.com/km4ege/eclipse2017_ionopath
