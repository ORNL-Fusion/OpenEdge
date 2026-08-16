# ##############################################################################
# This file sets common default options that all sparta builds use. These
# options can be overridden at configure time via `cmake -DVAR=VAL` or `cmake -C
# /path/to/preset/presets.cmake`
# ##############################################################################
set(SPARTA_DEFAULT_CXX_COMPILE_FLAGS
    -DSPARTA_GZIP
    CACHE
      STRING
      "Compiler flags used when building object files for the \"spa_\" executable"
)

set(SPARTA_MACHINE
    ""
    CACHE
      STRING
      "Suffix to append to spa binary (WON'T enable any features automatically)"
)

if(SPARTA_ENABLE_TESTING)
  set(SPARTA_ENABLED_TEST_SUITES
      # Small public verification suites. Larger scientific workflows are run
      # explicitly or through regression/run_regression.sh.
      "verification/collisions/coulomb"
      "verification/efield_polarization"
      "verification/ionization_recombination")

  set(SPARTA_DISABLED_TESTS "")

  list(APPEND __DEFAULT_MPI_RANKS "1")
  list(APPEND __DEFAULT_MPI_RANKS "4")
endif()
