.. hand-edited: this page is maintained here, not generated from doc/*.txt

2. Getting Started
==================

This chapter describes the contents of the distribution, how to build
and test the executable, how to run it, and how to read its output.

| 2.1 :ref:`What's in the OpenEdge distribution <start_1>`
| 2.2 :ref:`Building OpenEdge <start_2>`
| 2.3 :ref:`Testing OpenEdge <start_3>`
| 2.4 :ref:`Running OpenEdge <start_4>`
| 2.5 :ref:`Command-line options <start_5>`
| 2.6 :ref:`Screen output <start_6>`

----------

.. _start_1:

2.1 What's in the OpenEdge distribution
---------------------------------------

A checkout of the OpenEdge repository contains these files and
directories:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Entry
     - Contents
   * - README.md
     - overview and quick start
   * - LICENSE, LICENSE.SPARTA, NOTICES
     - the GNU General Public License version 2, the SPARTA license
       and the attributions of third-party code and data
   * - BUILD_CMAKE.md
     - build instructions, CMake options and recipes for specific
       machines
   * - bench
     - SPARTA benchmark problems
   * - cmake
     - the CMake project and the preset files
   * - data
     - SPARTA species, collision and reaction files
   * - database
     - the :ref:`processes.h5 <howto_6>` database of atomic and
       surface data, and the impact energy-angle distribution tables
   * - doc
     - this manual
   * - examples
     - verification and workflow cases, see
       :doc:`Section 5 <Section_example>`
   * - lib
     - the bundled Kokkos library
   * - Python
     - the SPARTA Python wrapper
   * - regression
     - the script that runs the registered example cases as tests
   * - scripts
     - environment set-up scripts for specific machines
   * - src
     - source files; the OpenEdge package is in src/OPENEDGE and the
       Kokkos versions of styles are in src/KOKKOS
   * - tools
     - plasma-background converters (tools/converters) and pre- and
       post-processing scripts

----------

.. _start_2:

2.2 Building OpenEdge
---------------------

OpenEdge is built with CMake in a directory outside the source tree.
The build needs

* a C++17 compiler,
* an MPI library, unless the serial preset is used,
* HDF5 with its C++ bindings,
* CMake 3.12 or later; the bundled Kokkos library needs CMake 3.16,
* optionally VTK 7.1 or later, for the VTK dump styles.

The OpenEdge commands form a package that is off by default and is
switched on with -DPKG_OPENEDGE=ON.  Without it the same source tree
builds plain SPARTA.  A build with MPI on Linux or macOS is

.. code-block:: text

   git clone https://github.com/ORNL-Fusion/OpenEdge.git
   mkdir buildOpenEdge
   cd buildOpenEdge
   cmake -C ../OpenEdge/cmake/presets/mpi.cmake ../OpenEdge/cmake -DPKG_OPENEDGE=ON
   make -j 8

The executable is written to the src sub-directory of the build
directory.  Its name is spa\_ followed by the value of the CMake
variable SPARTA_MACHINE, which each preset sets to its own name.  The
build above produces src/spa_mpi.

The preset files in cmake/presets are CMake cache files that set the
compilers, flags and packages for one configuration.  The ones used
most with OpenEdge are

.. list-table::
   :header-rows: 1
   :widths: 34 26 40

   * - Preset
     - Executable
     - Configuration
   * - mpi.cmake
     - spa_mpi
     - MPI compiler wrappers, -O3
   * - mac_mpi.cmake
     - spa_mac_mpi
     - MPI on macOS
   * - serial.cmake
     - spa_serial
     - no MPI library; the MPI stubs package is used
   * - mpi_big.cmake
     - spa_mpi_big
     - as mpi.cmake with 64-bit particle and cell IDs
       (-DSPARTA_BIGBIG)
   * - kokkos_omp.cmake
     - spa_kokkos_omp
     - Kokkos package with the OpenMP backend
   * - kokkos_cuda.cmake
     - spa_kokkos_cuda
     - Kokkos package with the CUDA backend for NVIDIA GPUs
   * - kokkos_cuda_perlmutter.cmake
     - spa_kokkos_cuda_perlmutter
     - Kokkos with CUDA for the A100 nodes of NERSC Perlmutter

The Kokkos presets switch the KOKKOS package on.  A GPU build is

.. code-block:: text

   cmake -C ../OpenEdge/cmake/presets/kokkos_cuda.cmake ../OpenEdge/cmake \
         -DPKG_OPENEDGE=ON -DKokkos_ARCH_AMPERE80=ON
   make -j 8

where the Kokkos_ARCH option names the GPU architecture.  A Kokkos
executable uses the Kokkos styles only when it is started with the
-kokkos and -suffix switches described in
:ref:`Section 2.5 <start_5>`.  Not every OpenEdge style has a Kokkos
version.  Compare a GPU result with a CPU run when you start a new
class of problem.

Other options are given on the cmake command line:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Effect
   * - -DPKG_OPENEDGE=ON
     - build the OpenEdge package (default OFF)
   * - -DPKG_KOKKOS=ON
     - build the KOKKOS package (default OFF)
   * - -DPKG_VTK=ON
     - build the *grid/vtk*, *surf/vtk* and *particle/vtk* dump
       styles; needs an installed VTK
   * - -DPKG_FFT=ON
     - build the FFT package
   * - -DBUILD_JPEG=ON, -DBUILD_PNG=ON
     - link the JPEG or PNG library for :doc:`dump image <dump_image>`
   * - -DSPARTA_MACHINE=name
     - name the executable spa_name

The command "cmake -LH ../OpenEdge/cmake" lists every option with its
help string.  To point CMake at a library that is not in a standard
location, set the environment variable <NAME>_ROOT, for example
HDF5_ROOT, before running cmake.  The BUILD_CMAKE.md file describes
the remaining options and gives recipes for specific machines.

The build records the absolute path of the database directory of the
source tree in the executable.  The executable finds
:ref:`processes.h5 <howto_6>` there unless the environment variable
OPENEDGE_ROOT points to another checkout.

----------

.. _start_3:

2.3 Testing OpenEdge
--------------------

The examples/verification directory holds small cases that compare
OpenEdge with an analytic solution, a reference run or another code.
Most have a run.sh script that runs the case, checks the result and
prints PASS or FAIL.  For example

.. code-block:: text

   cd examples/verification/pushers/orbit
   ./run.sh /path/to/spa_mpi

runs the Boris and guiding-centre orbit checks and exits with status 0
when every check passes.  The README file of each case gives its
commands and its acceptance criteria.

The script in the regression directory runs every registered case of
the examples directory for a limited number of steps:

.. code-block:: text

   ./regression/run_regression.sh --np 4 --exe /path/to/spa_mpi

A case passes if the run exits with status 0, its log file contains no
ERROR line, and, where the case directory holds a reference file, its
end-of-run metrics are within the stored tolerances.  The script takes
these options:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Meaning
   * - --np N
     - number of MPI processes (default 4)
   * - --exe PATH
     - executable to test
   * - --filter PATTERN
     - run only the cases whose name matches the glob pattern
   * - --nsteps N
     - number of steps per case (default 1000); workflow cases are
       always limited to 1000 steps
   * - --verbose
     - print the full output of a failed case
   * - --kk
     - run with the Kokkos styles; cases without Kokkos support are
       skipped
   * - --parity-cpu-exe PATH
     - with --kk, also compare the Kokkos executable with the CPU
       executable PATH on the examples/verification/kokkos_parity case
   * - --update-ref
     - write the reference files from this run

:doc:`Section 5 <Section_example>` describes the cases.

----------

.. _start_4:

2.4 Running OpenEdge
--------------------

OpenEdge reads its commands from an input script.  Give the script
with the -in switch, which works with every MPI launcher:

.. code-block:: text

   mpirun -np 4 /path/to/spa_mpi -in in.openedge

Redirection of standard input, as in "spa_mpi < in.openedge", also
works with most launchers.  :doc:`Section 3 <Section_commands>`
describes the structure of an input script and its commands.

File names in an input script are relative to the directory in which
the run is started, so a case is run from its own directory.  The
input scripts of the examples are named in.\*, read their data from
an input sub-directory and write to an output sub-directory.  Some
cases need a plasma background file that is produced with one of the
converters in tools/converters; the README of the case gives the
command.

At startup OpenEdge prints its version and the git revision it was
built from.  Keep this with the results of a run.  The log is written
to the file log.openedge unless the -log switch or the
:doc:`log <log>` command names another file.

When OpenEdge finds an error in the input script or during a run it
prints an ERROR message and stops.  For a condition that may be
harmless it prints a WARNING message and continues.
:doc:`Section 6 <Section_errors>` lists the messages.

A problem can be run on any number of processors, including one.  The
random-number streams differ from processor to processor, so two runs
of the same problem on different numbers of processors agree
statistically and not digit for digit.  Verification cases that follow
one particle are run on one processor for this reason.

The size of a problem is limited by the memory of the processors.  If
a run exhausts the memory, use more processors or reduce the number of
particles or grid cells.

----------

.. _start_5:

2.5 Command-line options
------------------------

At run time, OpenEdge recognizes several optional command-line switches
which may be used in any order.  Either the full word or a one-or-two
letter abbreviation can be used:

* -e or -echo
* -i or -in
* -h or -help
* -k or -kokkos
* -l or -log
* -p or -partition
* -pk or -package
* -pl or -plog
* -ps or -pscreen
* -sc or -screen
* -sf or -suffix
* -v or -var

For example, spa_mpi might be launched as follows:

.. parsed-literal::

   mpirun -np 16 spa_mpi -v f tmp.out -l my.log -sc none < in.openedge
   mpirun -np 16 spa_mpi -var f tmp.out -log my.log -screen none < in.openedge

Here are the details on the options:

.. parsed-literal::

   -echo style

Set the style of command echoing.  The style can be *none* or *screen*
or *log* or *both*.  Depending on the style, each command read from
the input script will be echoed to the screen and/or logfile.  This
can be useful to figure out which line of your script is causing an
input error.  The default value is *log*.  The echo style can also be
set by using the :doc:`echo <echo>` command in the input script itself.

.. parsed-literal::

   -in file

Specify a file to use as an input script.  This is an optional switch
when running OpenEdge in one-partition mode.  If it is not specified,
OpenEdge reads its input script from stdin, e.g. spa_mpi < in.run.
This is a required switch when running OpenEdge in multi-partition mode,
since multiple processors cannot all read from stdin.

.. parsed-literal::

   -help

Print the version and build configuration, the command-line switches,
and the list of styles (fix, compute, collide, etc) compiled into this
executable.  OpenEdge prints the information and exits immediately.

.. parsed-literal::

   -kokkos on/off keyword/value ...

Explicitly enable or disable KOKKOS support, as provided by the KOKKOS
package.  Even if OpenEdge is built with this package, as described
above in :ref:`Section 2.2 <start_2>`, this switch must be set to enable
running with the KOKKOS-enabled styles the package provides.  If the
switch is not set (the default), OpenEdge will operate as if the KOKKOS
package were not installed; i.e. you can run standard OpenEdge
for testing or benchmarking purposes.

Additional optional keyword/value pairs can be specified which
determine how Kokkos will use the underlying hardware on your
platform.  These settings apply to each MPI task you launch via the
"mpirun" or "mpiexec" command.  You may choose to run one or more MPI
tasks per physical node.  Note that if you are running on a desktop
machine, you typically have one physical node.  On a cluster or
supercomputer there may be dozens or 1000s of physical nodes.

Either the full word or an abbreviation can be used for the keywords.
Note that the keywords do not use a leading minus sign.  I.e. the
keyword is "t", not "-t".  Also note that each of the keywords has a
default setting.

* d or device
* g or gpus
* t or threads
* n or numa

.. parsed-literal::

   device Nd

This option is only relevant if you built OpenEdge with KOKKOS_DEVICES=Cuda, you
have more than one GPU per node, and if you are running with only one
MPI task per node.  The Nd setting is the ID of the GPU on the node to
run on.  By default Nd = 0.  If you have multiple GPUs per node, they
have consecutive IDs numbered as 0,1,2,etc.  This setting allows you
to launch multiple independent jobs on the node, each with a single
MPI task per node, and assign each job to run on a different GPU.

.. parsed-literal::

   gpus Ng Ns

This option is only relevant if you built OpenEdge with KOKKOS_DEVICES=Cuda, you
have more than one GPU per node, and you are running with multiple MPI
tasks per node.  The Ng setting is how many GPUs
you will use per node.  The Ns setting is optional.  If set, it is the ID of a
GPU to skip when assigning MPI tasks to GPUs.  This may be useful if
your desktop system reserves one GPU to drive the screen and the rest
are intended for computational work like running OpenEdge.  By default
Ng = 1 and Ns is not set.

Depending on which flavor of MPI you are running, OpenEdge will look for
one of these 4 environment variables

.. parsed-literal::

   SLURM_LOCALID (various MPI variants compiled with SLURM support)
   MPT_LRANK (HPE MPI)
   MV2_COMM_WORLD_LOCAL_RANK (Mvapich)
   OMPI_COMM_WORLD_LOCAL_RANK (OpenMPI)

which are initialized by the "srun", "mpirun" or "mpiexec" commands.
The environment variable setting for each MPI rank is used to assign a
unique GPU ID to the MPI task.

.. parsed-literal::

   threads Nt

This option assigns Nt number of threads to each MPI task for
performing work when Kokkos is executing in OpenMP or pthreads mode.
The default is Nt = 1, which essentially runs in MPI-only mode.  If
there are Np MPI tasks per physical node, you generally want Np\*Nt =
the number of physical cores per node, to use your available hardware
optimally. If OpenEdge is compiled with KOKKOS_DEVICES=Cuda,
this setting has no effect.

.. parsed-literal::

   -log file

Specify a log file for OpenEdge to write status information to.  In
one-partition mode, if the switch is not used, OpenEdge writes to the
file log.openedge.  If this switch is used, OpenEdge writes to the
specified file.  In multi-partition mode, if the switch is not used, a
log.openedge file is created with hi-level status information.  Each
partition also writes to a log.openedge.N file where N is the partition
ID.  If the switch is specified in multi-partition mode, the hi-level
logfile is named "file" and each partition also logs information to a
file.N.  For both one-partition and multi-partition mode, if the
specified file is "none", then no log files are created.  Using a
:doc:`log <log>` command in the input script will override this setting.
Option -plog will override the name of the partition log files file.N.

.. parsed-literal::

   -partition 8x2 4 5 ...

Invoke OpenEdge in multi-partition mode.  When OpenEdge is run on P
processors and this switch is not used, OpenEdge runs in one partition,
i.e. all P processors run a single simulation.  If this switch is
used, the P processors are split into separate partitions and each
partition runs its own simulation.  The arguments to the switch
specify the number of processors in each partition.  Arguments of the
form MxN mean M partitions, each with N processors.  Arguments of the
form N mean a single partition with N processors.  The sum of
processors in all partitions must equal P.  Thus the command
"-partition 8x2 4 5" has 10 partitions and runs on a total of 25
processors.  Note that with MPI installed on a machine (e.g. your
desktop), you can run on more (virtual) processors than you have
physical processors.

To run multiple independent simulations from one input script, using
multiple partitions, see :ref:`Section 4.7 <howto_7>` of
the manual.  World- and universe-style variables are useful in this
context.

.. parsed-literal::

   -package style args ....

Invoke the :doc:`package <package>` command with style and args.  The
syntax is the same as if the command appeared at the top of the input
script.  For example "-package kokkos on gpus 2" or "-pk kokkos g 2"
is the same as :doc:`package kokkos g 2 <package>` in the input
script.  The possible styles
and args are documented on the :doc:`package <package>` doc page.  This
switch can be used multiple times.

Along with the "-suffix" command-line switch, this is a convenient
mechanism for invoking the KOKKOS accelerator package and its options without
having to edit an input script.

.. parsed-literal::

   -plog file

Specify the base name for the partition log files, so partition N
writes log information to file.N. If file is none, then no partition
log files are created.  This overrides the filename specified in the
-log command-line option.  This option is useful when working with
large numbers of partitions, allowing the partition log files to be
suppressed (-plog none) or placed in a sub-directory (-plog
replica_files/log.openedge) If this option is not used the log file for
partition N is log.openedge.N or whatever is specified by the -log
command-line option.

.. parsed-literal::

   -pscreen file

Specify the base name for the partition screen file, so partition N
writes screen information to file.N. If file is none, then no
partition screen files are created.  This overrides the filename
specified in the -screen command-line option.  This option is useful
when working with large numbers of partitions, allowing the partition
screen files to be suppressed (-pscreen none) or placed in a
sub-directory (-pscreen replica_files/screen) If this option is not
used the screen file for partition N is screen.N or whatever is
specified by the -screen command-line option.

.. parsed-literal::

   -screen file

Specify a file for OpenEdge to write its screen information to.  In
one-partition mode, if the switch is not used, OpenEdge writes to the
screen.  If this switch is used, OpenEdge writes to the specified file
instead and you will see no screen output.  In multi-partition mode,
if the switch is not used, hi-level status information is written to
the screen.  Each partition also writes to a screen.N file where N is
the partition ID.  If the switch is specified in multi-partition mode,
the hi-level screen dump is named "file" and each partition also
writes screen information to a file.N.  For both one-partition and
multi-partition mode, if the specified file is "none", then no screen
output is performed. Option -pscreen will override the name of the
partition screen files file.N.

.. parsed-literal::

   -suffix style args

Use variants of various styles if they exist.  The specified style can
be *kk*.  This refers to the optional KOKKOS package that OpenEdge can
be built with, as described above in :ref:`Section 2.2 <start_2>`.

Along with the "-package" command-line switch, this is a convenient
mechanism for invoking the KOKKOS accelerator package and its options without
having to edit an input script.

As an example, the KOKKOS package provides a
:doc:`compute_style temp <compute_temp>` variant, with style name
temp/kk. A variant style
can be specified explicitly in your input script, e.g. compute
temp/kk. If the suffix command is used with the appropriate style,
you do not need to modify your input script.  The specified suffix
(kk) is automatically appended whenever your
input script command creates a new :doc:`fix <fix>`,
:doc:`compute <compute>`, etc.
If the variant version does not exist, the standard version is
created.

For the KOKKOS package, using this command-line switch also invokes
the default KOKKOS settings, as if the command "package kokkos" were
used at the top of your input script.  These settings can be changed
by using the "-package kokkos" command-line switch or the
:doc:`package kokkos <package>` command in your script.

The :doc:`suffix <suffix>` command can also be used within an input
script to set a suffix, or to turn off or back on any suffix setting
made via the command line.

.. parsed-literal::

   -var name value1 value2 ...

Specify a variable that will be defined for substitution purposes when
the input script is read.  "Name" is the variable name which can be a
single character (referenced as $x in the input script) or a full
string (referenced as ${abc}).  An
:doc:`index-style variable <variable>` will be created and populated with the
subsequent values, e.g. a set of filenames.  Using this command-line
option is equivalent to putting the line "variable name index value1
value2 ..."  at the beginning of the input script.  Defining an index
variable as a command-line argument overrides any setting for the same
index variable in the input script, since index variables cannot be
re-defined.  See the :doc:`variable <variable>` command for more info on
defining index and other kinds of variables and
:ref:`Section 3.2 <cmd_2>` for more info on using variables in input
scripts.

.. warning::

   Currently, the command-line parser looks for arguments
   that start with "-" to indicate new switches. Thus you cannot specify
   multiple variable values if any of they start with a "-", e.g. a
   negative numeric value. It is OK if the first value1 starts with a
   "-", since it is automatically skipped.

----------

.. _start_6:

2.6 Screen output
-----------------

As OpenEdge reads an input script, it prints information to both the
screen and a log file about significant actions it takes to setup a
simulation.  When the simulation is ready to begin, OpenEdge performs
various initializations and prints the amount of memory (in MBytes per
processor) that the simulation requires.  It also prints details of
the initial state of the system.  During the run itself, statistical
information is printed periodically, every few timesteps.  When the
run concludes, OpenEdge prints the final state and a total run time for
the simulation.  It then appends statistics about the CPU time and
size of information stored for the simulation.  An example set of
statistics is shown here:

.. parsed-literal::

   Loop time of 0.639973 on 4 procs for 1000 steps with 45792 particles

   MPI task timing breakdown:
   Section \|  min time  \|  avg time  \|  max time  \|%varavg\| %total
   ---------------------------------------------------------------
   Move    \| 0.10948    \| 0.26191    \| 0.42049    \|  27.6 \| 40.92
   Coll    \| 0.013711   \| 0.041659   \| 0.070985   \|  13.5 \|  6.51
   Sort    \| 0.01733    \| 0.040286   \| 0.063573   \|  10.6 \|  6.29
   Comm    \| 0.02276    \| 0.023555   \| 0.02493    \|   0.6 \|  3.68
   Modify  \| 0.00018167 \| 0.024758   \| 0.051345   \|  15.6 \|  3.87
   Output  \| 0.0002172  \| 0.0007354  \| 0.0012152  \|   0.0 \|  0.11
   Other   \|            \| 0.2471     \|            \|       \| 38.61

.. parsed-literal::

   Particle moves    = 38096354 (38.1M)
   Cells touched     = 43236871 (43.2M)
   Particle comms    = 146623 (0.147M)
   Boundary collides = 182782 (0.183M)
   Boundary exits    = 181792 (0.182M)
   SurfColl checks   = 7670863 (7.67M)
   SurfColl occurs   = 177740 (0.178M)
   Surf reactions    = 124169 (0.124M)
   Collide attempts  = 1232 (1K)
   Collide occurs    = 553 (0.553K)
   Gas reactions     = 23 (0.023K)
   Particles stuck   = 0

.. parsed-literal::

   Particle-moves/CPUsec/proc: 1.4882e+07
   Particle-moves/step: 38096.4
   Cell-touches/particle/step: 1.13493
   Particle comm iterations/step: 1.999
   Particle fraction communicated: 0.00384874
   Particle fraction colliding with boundary: 0.00479789
   Particle fraction exiting boundary: 0.0047719
   Surface-checks/particle/step: 0.201354
   Surface-collisions/particle/step: 0.00466554
   Surface-reactions/particle/step: 0.00325934
   Collision-attempts/particle/step: 1.232
   Collisions/particle/step: 0.553
   Gas-reactions/particle/step: 0.023

.. parsed-literal::

   Gas reaction tallies:
     style tce #-of-reactions 45
     reaction O2 + N --> O + O + N: 10
     reaction O2 + O --> O + O + O: 5
     reaction N2 + O --> N + N + O: 8

.. parsed-literal::

   Surface reaction tallies:
     id 1 style global #-of-reactions 2
       reaction all: 124025
       reaction delete: 53525
       reaction create: 70500

.. parsed-literal::

   Particles: 11448 ave 17655 max 5306 min
   Histogram: 2 0 0 0 0 0 0 0 0 2
   Cells:     100 ave 100 max 100 min
   Histogram: 4 0 0 0 0 0 0 0 0 0
   GhostCell: 21 ave 21 max 21 min
   Histogram: 4 0 0 0 0 0 0 0 0 0
   EmptyCell: 21 ave 21 max 21 min
   Histogram: 4 0 0 0 0 0 0 0 0 0
   Surfs:     50 ave 50 max 50 min
   Histogram: 4 0 0 0 0 0 0 0 0 0
   GhostSurf: 0 ave 0 max 0 min
   Histogram: 4 0 0 0 0 0 0 0 0 0

The first line gives the total CPU run time for the simulation, in
seconds.

The next section gives a breakdown of the CPU timing (in seconds) in
7 categories.  The first four are timings for particles moves, which
includes interaction with surface elements, then particle collisions,
then sorting of particles (required to perform collisions), and
communication of particles between processors.  The Modify section is
time for operations invoked by fixes and computes.  The Output section
is for dump command and statistical output.  The Other category is
typically for load-imbalance, as some MPI tasks wait for others MPI
tasks to complete.  In each category the min,ave,max time across
processors is shown, as well as a variation, and the percentage of
total time.

The next section gives some statistics about the run.  These are total
counts of particle moves, grid cells touched by particles, the number
of particles communicated between processors, collisions of particles
with the global boundary and with surface elements (none in this
problem), as well as collision and reaction statistics.

The next section gives additional statistics, normalized by timestep
or processor count.

The next 2 sections are optional.  The "Gas reaction tallies" section
is only output if the :doc:`react <react>` command is used.  For each
reaction with a non-zero tally, the number of those reactions that
occurred during the run is printed.  The "Surface reaction tallies"
section is only output if the :doc:`surf_react <surf_react>` command was
used one or more times, to assign reaction models to individual
surface elements or the box boundaries.  For each of the commands, and
each of its reactions with a non-zero tally, the number of those
reactions that occurred during the run is printed.  Note that this is
effectively a summation over all the surface elements and/or box
boundaries the :doc:`surf_react <surf_react>` command was used to assign
a reaction model to.

The last section is a histogramming across processors of various
per-processor statistics: particle count, owned grid cells, processor,
ghost grid cells which are copies of cells owned by other processors,
and empty cells which are ghost cells without surface information
(only used to pass particles to neighboring processors).

The ave value is the average across all processors.  The max and min
values are for any processor.  The 10-bin histogram shows the
distribution of the value across processors.  The total number of
histogram counts is equal to the number of processors.
