/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.sandia.gov
   Steve Plimpton, sjplimp@gmail.com, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */

#ifdef DUMP_CLASS

DumpStyle(tally,DumpTally)

#else

#ifndef SPARTA_DUMP_TALLY_H
#define SPARTA_DUMP_TALLY_H

#include "dump.h"

namespace SPARTA_NS {

class DumpTally : public Dump {
 public:
  DumpTally(class SPARTA *, int, char **);
  ~DumpTally();

 private:
  int nevery;                // dump frequency to check Fix against

  char *columns;             // column labels

  int nfield;                // # of keywords listed by user
  int ioptional;             // index of start of optional args

  int *field2index;          // which compute,fix,variable calcs this field
  int *argindex;             // index into compute,fix scalar_atom,vector_atom
                             // 0 for scalar_atom, 1-N for vector_atom values
  int ncompute;              // # of Compute objects used by dump
  char **id_compute;         // their IDs
  class Compute **compute;   // list of ptrs to the Compute objects

  int dimension;
  int ntally;                // # of tallies output by this proc

  // private methods

  void init_style();
  void write_header(bigint);
  int count();
  void pack();
  void write_data(int, double *);

  int parse_fields(int, char **);
  int add_compute(char *);

  typedef void (DumpTally::*FnPtrHeader)(bigint);
  FnPtrHeader header_choice;           // ptr to write header functions
  void header_binary(bigint);
  void header_item(bigint);

  typedef void (DumpTally::*FnPtrData)(int, double *);
  FnPtrData write_choice;              // ptr to write data functions
  void write_binary(int, double *);
  void write_string(int, double *);
  void write_text(int, double *);

  // customize by adding a method prototype

  typedef void (DumpTally::*FnPtrPack)(int);
  FnPtrPack *pack_choice;              // ptrs to pack functions

  void pack_compute(int);

  // union data struct for packing 32-bit and 64-bit ints into double bufs
  // this avoids aliasing issues by having 3 pointers (double,int,uint)
  //   to same buf memory
  // constructor for 32-bit int or uint prevents compiler
  //   from possibly calling the double constructor when passed an int/uint
  // copy to a double *buf:
  //   buf[m++] = ubuf(foo).d, where foo is a 32-bit or 64-bit int or uint
  // copy from a double *buf:
  //   foo = (int) ubuf(buf[m++]).i or foo = (uint64_t) ubuf(buf[m++]).u
  //   the cast prevents compiler warnings about possible truncation

  union ubuf {
    double d;
    int64_t i;
    uint64_t u;
    ubuf(double arg) : d(arg) {}
    ubuf(int64_t arg) : i(arg) {}
    ubuf(int arg) : i(arg) {}
    ubuf(uint64_t arg) : u(arg) {}
    ubuf(uint32_t arg) : u(arg) {}
  };
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: No dump surf attributes specified

Self-explanatory.

E: Invalid attribute in dump surf command

Self-explanatory.

E: Could not find dump surf compute ID

Self-explanatory.

E: Could not find dump surf fix ID

Self-explanatory.

E: Dump surf and fix not computed at compatible times

Fixes generate values on specific timesteps.  The dump surf output
does not match these timesteps.

E: Could not find dump surf variable name

Self-explanatory.

E: Invalid dump surf field for 2d simulation

Self-explanatory.

E: Dump surf compute does not compute per-surf info

Self-explanatory.

E: Dump surf compute does not calculate per-surf array

Self-explanatory.

E: Dump surf compute vector is accessed out-of-range

Self-explanatory.

E: Dump surf fix does not compute per-surf info

Self-explanatory.

E: Dump surf fix does not compute per-surf array

Self-explanatory.

E: Dump surf fix vector is accessed out-of-range

Self-explanatory.

E: Dump surf variable is not surf-style variable

Self-explanatory.

*/
