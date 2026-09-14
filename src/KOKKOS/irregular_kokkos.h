/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.github.io
   Steve Plimpton, sjplimp@gmail.com, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */

#ifndef SPARTA_IRREGULAR_KOKKOS_H
#define SPARTA_IRREGULAR_KOKKOS_H

#include "irregular.h"
#include "kokkos_type.h"

namespace SPARTA_NS {

struct TagIrregularPackBuffer{};

struct TagIrregularUnpackBufferSelf{};

class IrregularKokkos : public Irregular {
 public:

  IrregularKokkos(class SPARTA *);
  ~IrregularKokkos();
  int create_data_uniform(int, int *, int sort = 0);
  int augment_data_uniform(int, int *);
  void exchange_uniform(DAT::t_char_1d, int, char *, DAT::t_char_1d);
  int create_data_uniform_flag(int, int *, int flag_in, int &flag_out);   // one MPI_Alltoall, carries a flag max
  // OE_COMM_TIMING accumulators (s): 0 irecv post, 1 pack kernel, 2 send (+D2H),
  // 3 self copy, 4 waitall, 5 recv H2D
  double oe_xt[6];

  KOKKOS_INLINE_FUNCTION
  void operator()(TagIrregularPackBuffer, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagIrregularUnpackBufferSelf, const int&) const;

  inline
  void pack_buffer_serial(const int, const int) const;

 private:
  int offset_send;
  int *oe_a2a_s,*oe_a2a_r;      // [2*nprocs] (count, flag) blocks for create_data_uniform_flag

  DAT::tdual_int_1d k_index_send;
  DAT::t_int_1d d_index_send;
  DAT::tdual_int_1d k_index_self;
  DAT::t_int_1d d_index_self;

  DAT::t_char_1d d_sendbuf;
  DAT::t_char_1d d_recvbuf;
  DAT::t_char_1d d_buf;
  HAT::t_char_1d h_recvbuf;
  HAT::t_char_1d h_sendbuf;     // OpenEdge perf: persistent self-unpack mirror
  HAT::t_char_1d h_buf;
  int nbytes;
};

}

#endif

/* ERROR/WARNING messages:

*/
