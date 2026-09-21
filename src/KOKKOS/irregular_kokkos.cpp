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

#include "spatype.h"
#include "mpi.h"
#include "stdlib.h"
#include "string.h"
#include "irregular_kokkos.h"
#include "particle.h"
#include "domain.h"
#include "comm.h"
#include "memory_kokkos.h"
#include "error.h"
#include "kokkos.h"



// DEBUG
#include "update.h"
#include "grid.h"

using namespace SPARTA_NS;

// allocate space for static class variable
// prototype for non-class function

//int *IrregularKokkos::proc_recv_copy;
int compare_standalone(const void *, const void *);

#define BUFFACTOR 1.5
#define BUFMIN 1000
#define BUFEXTRA 1000

/* ---------------------------------------------------------------------- */

IrregularKokkos::IrregularKokkos(SPARTA *sparta) : Irregular(sparta)
{
  for (int i = 0; i < 6; i++) oe_xt[i] = 0.0;
  memory->create(oe_a2a_s,2*nprocs,"irregular:a2a_s");
  memory->create(oe_a2a_r,2*nprocs,"irregular:a2a_r");
  // plan collective: an Allreduce of nprocs*nprocs+1 ints (shared-memory tree on a
  // node) up to 32 ranks, an Alltoall of 2*nprocs ints above; OE_PLAN_COLL overrides
  oe_plan_coll = (nprocs <= 32) ? 1 : 0;
  if (const char *e = getenv("OE_PLAN_COLL")) oe_plan_coll = (strcmp(e,"allreduce") == 0) ? 1 : 0;
  if (nprocs > 32) oe_plan_coll = 0;
  oe_ar_buf = NULL;
  if (oe_plan_coll) memory->create(oe_ar_buf,nprocs*nprocs+1,"irregular:ar_buf");
}

/* ---------------------------------------------------------------------- */

IrregularKokkos::~IrregularKokkos()
{
  if (copymode) return;

  memoryKK->destroy_kokkos(k_index_send,index_send);
  index_send = NULL;

  memoryKK->destroy_kokkos(k_index_self,index_self);
  index_self = NULL;
  memory->destroy(oe_a2a_s);
  memory->destroy(oe_a2a_r);
  memory->destroy(oe_ar_buf);
}

/* ----------------------------------------------------------------------
   OpenEdge (2026-09-14): create_data_uniform with ONE collective. An
   MPI_Alltoall of (count, flag) per destination replaces the
   Reduce_scatter, the per-destination count Send/Recv with MPI_ANY_SOURCE
   and the trailing MPI_Barrier of create_data_uniform, and carries the
   mover's per-pass entry/exit flag (flag_out = max over ranks) so a move
   pass costs one collective instead of four. Received messages are always
   ordered by source rank (a superset of the sort option).
   return total # of datums I will recv, including any to self
------------------------------------------------------------------------- */

int IrregularKokkos::create_data_uniform_flag(int n, int *proclist, int flag_in, int &flag_out)
{
  int i,m;

  for (i = 0; i < nprocs; i++) work1[i] = 0;
  for (i = 0; i < n; i++) work1[proclist[i]]++;
  if (oe_plan_coll) {
    // Allreduce variant: row me = my counts per destination, last slot = flag; after
    // the SUM every rank holds the full count matrix and the flag sum (> 0 = any)
    const int nn = nprocs*nprocs;
    for (i = 0; i < nn+1; i++) oe_ar_buf[i] = 0;
    for (i = 0; i < nprocs; i++) oe_ar_buf[me*nprocs+i] = (i == me) ? 0 : work1[i];
    oe_ar_buf[nn] = flag_in ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE,oe_ar_buf,nn+1,MPI_INT,MPI_SUM,world);
    for (i = 0; i < nprocs; i++) {
      oe_a2a_r[2*i] = oe_ar_buf[i*nprocs+me];
      oe_a2a_r[2*i+1] = oe_ar_buf[nn] > 0 ? 1 : 0;
    }
  } else {
    for (i = 0; i < nprocs; i++) {
      oe_a2a_s[2*i] = (i == me) ? 0 : work1[i];
      oe_a2a_s[2*i+1] = flag_in;
    }
    MPI_Alltoall(oe_a2a_s,2,MPI_INT,oe_a2a_r,2,MPI_INT,world);
  }

  // receive side: procs sending to me, ascending rank order, and the flag max

  nrecv = 0;
  nrecvdatum = 0;
  flag_out = flag_in;
  for (i = 0; i < nprocs; i++) {
    if (oe_a2a_r[2*i+1] > flag_out) flag_out = oe_a2a_r[2*i+1];
    if (i == me || oe_a2a_r[2*i] == 0) continue;
    proc_recv[nrecv] = i;
    num_recv[nrecv] = oe_a2a_r[2*i];
    nrecvdatum += num_recv[nrecv];
    nrecv++;
  }

  // send side: same bookkeeping as create_data_uniform

  nsend = 0;
  for (i = 0; i < nprocs; i++)
    if (work1[i]) nsend++;
  if (work1[me]) nsend--;

  if (n > indexmax) {
    indexmax = n;
    memoryKK->destroy_kokkos(k_index_send,index_send);
    memoryKK->create_kokkos(k_index_send,index_send,indexmax,"irregular:index_send");
    d_index_send = k_index_send.view_device();
  }
  if (work1[me] > indexselfmax) {
    indexselfmax = work1[me];
    memoryKK->destroy_kokkos(k_index_self,index_self);
    memoryKK->create_kokkos(k_index_self,index_self,indexselfmax,"irregular:index_self");
    d_index_self = k_index_self.view_device();
  }

  int iproc = me;
  int isend = 0;
  for (i = 0; i < nprocs; i++) {
    iproc++;
    if (iproc == nprocs) iproc = 0;
    if (iproc == me) {
      num_self = work1[iproc];
      work1[iproc] = 0;
    } else if (work1[iproc]) {
      proc_send[isend] = iproc;
      num_send[isend] = work1[iproc];
      work1[iproc] = isend;
      isend++;
    }
  }
  work2[0] = 0;
  for (i = 1; i < nsend; i++) work2[i] = work2[i-1] + num_send[i-1];
  m = 0;
  for (i = 0; i < n; i++) {
    iproc = proclist[i];
    if (iproc == me) index_self[m++] = i;
    else {
      isend = work1[iproc];
      index_send[work2[isend]++] = i;
    }
  }
  k_index_self.modify_host();
  k_index_send.modify_host();

  sendmax = 0;
  for (i = 0; i < nsend; i++) sendmax = MAX(sendmax,num_send[i]);
  nrecvdatum += num_self;
  for (i = 0; i < nrecv; i++) proc2recv[proc_recv[i]] = i;
  return nrecvdatum;
}

/* ----------------------------------------------------------------------
   create communication plan based on list of datums of uniform size
   n = # of datums to send
   proclist = proc to send each datum to, can include self
   sort = flag for sorting order of received datums by proc ID
   return total # of datums I will recv, including any to self
------------------------------------------------------------------------- */

int IrregularKokkos::create_data_uniform(int n, int *proclist, int sort)
{
  int i,m;

  // setup for collective comm
  // work1 = # of datums I send to each proc, set self to 0
  // work2 = 1 for all procs, used for ReduceScatter

  for (i = 0; i < nprocs; i++) {
    work1[i] = 0;
    work2[i] = 1;
  }
  for (i = 0; i < n; i++) work1[proclist[i]] = 1;
  work1[me] = 0;

  // nrecv = # of procs I receive messages from, not including self
  // options for performing ReduceScatter operation
  // some are more efficient on some machines at big sizes

#ifdef SPARTA_RS_ALLREDUCE_INPLACE
  MPI_Allreduce(MPI_IN_PLACE,work1,nprocs,MPI_INT,MPI_SUM,world);
  nrecv = work1[me];
#else
#ifdef SPARTA_RS_ALLREDUCE
  MPI_Allreduce(work1,work2,nprocs,MPI_INT,MPI_SUM,world);
  nrecv = work2[me];
#else
  MPI_Reduce_scatter(work1,&nrecv,work2,MPI_INT,MPI_SUM,world);
#endif
#endif

  // work1 = # of datums I send to each proc, including self
  // nsend = # of procs I send messages to, not including self

  for (i = 0; i < nprocs; i++) work1[i] = 0;
  for (i = 0; i < n; i++) work1[proclist[i]]++;

  nsend = 0;
  for (i = 0; i < nprocs; i++)
    if (work1[i]) nsend++;
  if (work1[me]) nsend--;

  // reallocate send and self index lists if necessary
  // could use n-work1[me] for length of index_send to be more precise

  if (n > indexmax) {
    indexmax = n;
    memoryKK->destroy_kokkos(k_index_send,index_send);
    memoryKK->create_kokkos(k_index_send,index_send,indexmax,"irregular:index_send");
    d_index_send = k_index_send.view_device();
  }

  if (work1[me] > indexselfmax) {
    indexselfmax = work1[me];
    memoryKK->destroy_kokkos(k_index_self,index_self);
    memoryKK->create_kokkos(k_index_self,index_self,indexselfmax,"irregular:index_self");
    d_index_self = k_index_self.view_device();
  }

  // proc_send = procs I send to
  // num_send = # of datums I send to each proc
  // num_self = # of datums I copy to self
  // to balance pattern of send messages:
  //   each proc starts with iproc > me, continues until iproc = me
  // reset work1 to store which send message each proc corresponds to

  int iproc = me;
  int isend = 0;
  for (i = 0; i < nprocs; i++) {
    iproc++;
    if (iproc == nprocs) iproc = 0;
    if (iproc == me) {
      num_self = work1[iproc];
      work1[iproc] = 0;
    } else if (work1[iproc]) {
      proc_send[isend] = iproc;
      num_send[isend] = work1[iproc];
      work1[iproc] = isend;
      isend++;
    }
  }

  // work2 = offsets into index_send for each proc I send to
  // m = ptr into index_self
  // index_send = list of which datums to send to each proc
  //   1st N1 values are datum indices for 1st proc,
  //   next N2 values are datum indices for 2nd proc, etc
  // index_self = list of which datums to copy to self

  work2[0] = 0;
  for (i = 1; i < nsend; i++) work2[i] = work2[i-1] + num_send[i-1];

  m = 0;
  for (i = 0; i < n; i++) {
    iproc = proclist[i];
    if (iproc == me) index_self[m++] = i;
    else {
      isend = work1[iproc];
      index_send[work2[isend]++] = i;
    }
  }
  k_index_self.modify_host();
  k_index_send.modify_host();

  // tell receivers how many datums I send them
  // sendmax = largest # of datums I send in a single message

  sendmax = 0;
  for (i = 0; i < nsend; i++) {
    MPI_Send(&num_send[i],1,MPI_INT,proc_send[i],0,world);
    sendmax = MAX(sendmax,num_send[i]);
  }

  // receive incoming messages
  // proc_recv = procs I recv from
  // num_recv = # of datums each proc sends me
  // nrecvdatum = total # of datums I recv

  nrecvdatum = 0;
  for (i = 0; i < nrecv; i++) {
    MPI_Recv(&num_recv[i],1,MPI_INT,MPI_ANY_SOURCE,0,world,status);
    proc_recv[i] = status->MPI_SOURCE;
    nrecvdatum += num_recv[i];
  }
  nrecvdatum += num_self;

  // sort proc_recv and num_recv by proc ID if requested
  // useful for debugging to insure reproducible ordering of received datums

  if (sort) {
    int *order = new int[nrecv];
    int *proc_recv_ordered = new int[nrecv];
    int *num_recv_ordered = new int[nrecv];

    for (i = 0; i < nrecv; i++) order[i] = i;
    proc_recv_copy = proc_recv;
    qsort(order,nrecv,sizeof(int),compare_standalone);

    int j;
    for (i = 0; i < nrecv; i++) {
      j = order[i];
      proc_recv_ordered[i] = proc_recv[j];
      num_recv_ordered[i] = num_recv[j];
    }

    memcpy(proc_recv,proc_recv_ordered,nrecv*sizeof(int));
    memcpy(num_recv,num_recv_ordered,nrecv*sizeof(int));
    delete [] order;
    delete [] proc_recv_ordered;
    delete [] num_recv_ordered;
  }

  // proc2recv[I] = which recv the Ith proc ID is
  // will only be accessed by procs I actually receive from

  for (i = 0; i < nrecv; i++) proc2recv[proc_recv[i]] = i;

  // barrier to insure all MPI_ANY_SOURCE messages are received
  // else another proc could proceed to exchange_data() and send to me

  MPI_Barrier(world);

  // return # of datums I will receive

  return nrecvdatum;
}

/* ----------------------------------------------------------------------
   augment communication plan with new datums of uniform size
   called after create_procs() created initial plan
   n = # of datums to send
   proclist = proc to send each datum to, can include self
   return total # of datums I will recv
------------------------------------------------------------------------- */

int IrregularKokkos::augment_data_uniform(int n, int *proclist)
{
  int i,m,iproc,isend;

  // tally count of messages to each proc in num_send and num_self

  num_self = 0;
  for (i = 0; i < nsend; i++) work2[proc_send[i]] = 0;
  work2[me] = 0;
  for (i = 0; i < n; i++) work2[proclist[i]]++;
  for (i = 0; i < nsend; i++) num_send[i] = work2[proc_send[i]];
  num_self = work2[me];

  // reallocate send and self index lists if necessary
  // could use n-num_self for length of index_send to be more precise

  if (n > indexmax) {
    indexmax = n;
    memoryKK->destroy_kokkos(k_index_send,index_send);
    memoryKK->create_kokkos(k_index_send,index_send,indexmax,"irregular:index_send");
    d_index_send = k_index_send.view_device();
  }

  if (num_self > indexselfmax) {
    indexselfmax = num_self;
    memoryKK->destroy_kokkos(k_index_self,index_self);
    memoryKK->create_kokkos(k_index_self,index_self,indexselfmax,"irregular:index_self");
    d_index_self = k_index_self.view_device();
  }

  // work2 = offsets into index_send for each proc I send to
  // m = ptr into index_self
  // index_send = list of which datums to send to each proc
  //   1st N1 values are datum indices for 1st proc,
  //   next N2 values are datum indices for 2nd proc, etc
  // index_self = list of which datums to copy to self

  work2[0] = 0;
  for (i = 1; i < nsend; i++) work2[i] = work2[i-1] + num_send[i-1];

  if (num_self) {
    m = 0;
    for (i = 0; i < n; i++) {
      iproc = proclist[i];
      if (iproc == me) index_self[m++] = i;
      else {
        isend = work1[iproc];
        index_send[work2[isend]++] = i;
      }
    }
    k_index_self.modify_host();
    k_index_send.modify_host();
  } else {
    for (i = 0; i < n; i++) {
      isend = work1[proclist[i]];
      index_send[work2[isend]++] = i;
    }
    k_index_send.modify_host();
  }

  // tell receivers how many datums I send them
  // sendmax = largest # of datums I send in a single message

  sendmax = 0;
  for (i = 0; i < nsend; i++) {
    MPI_Send(&num_send[i],1,MPI_INT,proc_send[i],0,world);
    sendmax = MAX(sendmax,num_send[i]);
  }

  // receive incoming messages
  // num_recv = # of datums each proc sends me
  // nrecvdatum = total # of datums I recv

  nrecvdatum = 0;
  for (i = 0; i < nrecv; i++) {
    MPI_Recv(&m,1,MPI_INT,MPI_ANY_SOURCE,0,world,status);
    iproc = status->MPI_SOURCE;
    num_recv[proc2recv[iproc]] = m;
    nrecvdatum += m;
  }
  nrecvdatum += num_self;

  // barrier to insure all MPI_ANY_SOURCE messages are received
  // else another proc could proceed to exchange_data() and send to me

  MPI_Barrier(world);

  // return # of datums I will receive

  return nrecvdatum;
}

/* ----------------------------------------------------------------------
   communicate uniform-size datums via existing plan
   sendbuf = list of datums to send
   nbytes = size of each datum
   recvbuf = received datums, including copied from me
------------------------------------------------------------------------- */

void IrregularKokkos::exchange_uniform(DAT::t_char_1d d_sendbuf_in, int nbytes_in,
                                       char* d_recvbuf_ptr, DAT::t_char_1d d_recvbuf_in)
{
  nbytes = nbytes_in;
  d_sendbuf = d_sendbuf_in;
  d_recvbuf = d_recvbuf_in;

  if (!sparta->kokkos->gpu_aware_flag &&
      h_recvbuf.extent(0) < d_recvbuf.extent(0)) {   // OpenEdge perf: grow-only mirror
    h_recvbuf = HAT::t_char_1d(Kokkos::view_alloc("irregular:d_recvbuf:mirror",Kokkos::WithoutInitializing),d_recvbuf.extent(0));
  }

  // post all receives, starting after self copies

  double oe_t = MPI_Wtime();
  bigint offset = (bigint)num_self*nbytes;
  for (int irecv = 0; irecv < nrecv; irecv++) {
    if (sparta->kokkos->gpu_aware_flag) {
      MPI_Irecv(&d_recvbuf_ptr[offset],num_recv[irecv]*nbytes,MPI_CHAR,
                proc_recv[irecv],0,world,&request[irecv]);
    } else {
      MPI_Irecv(h_recvbuf.data() + offset,num_recv[irecv]*nbytes,MPI_CHAR,
                proc_recv[irecv],0,world,&request[irecv]);
    }
    offset += (bigint)num_recv[irecv]*nbytes;
  }

  oe_xt[0] += MPI_Wtime() - oe_t;

  // OpenEdge perf (2026-09-14): one gather kernel for all destinations (index_send
  // is already grouped by destination), one fence, one D2H copy when host-staged,
  // then one MPI_Send per destination from the packed offsets. Upstream launched a
  // pack kernel + fence (+ D2H) per destination: ~3 migrate calls per step x up to
  // nprocs-1 destinations of launch/fence latency was most of the Comm bucket.

  int total_send = 0;
  for (int isend = 0; isend < nsend; isend++) total_send += num_send[isend];
  const bigint need = (bigint)total_send*nbytes;
  if (need > MAXSMALLINT)
    error->one(FLERR,"Irregular comm send buffer exceeds 2 GB, try using"
                     "'global mem/limit' command");
  if (need > (bigint)d_buf.extent(0)) {   // grow-only
    d_buf = DAT::t_char_1d(Kokkos::view_alloc("irregular:buf",Kokkos::WithoutInitializing),need);
    if (!sparta->kokkos->gpu_aware_flag)
      h_buf = HAT::t_char_1d(Kokkos::view_alloc("irregular:buf:mirror",Kokkos::WithoutInitializing),need);
  }
  bufmax = (int) d_buf.extent(0);

  if (sparta->kokkos->gpu_aware_flag)
    k_index_self.sync_device();

  k_index_send.sync_device();

  oe_t = MPI_Wtime();
  offset_send = 0;
  if (total_send) {
    copymode = 1;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagIrregularPackBuffer>(0,total_send),*this);
    copymode = 0;
    if (sparta->kokkos->gpu_aware_flag) DeviceType().fence();
    else Kokkos::deep_copy(Kokkos::subview(h_buf,std::make_pair((bigint)0,need)),
                           Kokkos::subview(d_buf,std::make_pair((bigint)0,need)));   // fences
  }
  oe_xt[1] += MPI_Wtime() - oe_t; oe_t = MPI_Wtime();

  {
    bigint off = 0;
    char *src = sparta->kokkos->gpu_aware_flag ? d_buf.data() : h_buf.data();
    for (int isend = 0; isend < nsend; isend++) {
      const int count = num_send[isend];
      MPI_Send(src + off,count*nbytes,MPI_CHAR,proc_send[isend],0,world);
      off += (bigint)count*nbytes;
    }
  }
  oe_xt[2] += MPI_Wtime() - oe_t;

  // copy datums to self, put at beginning of recvbuf

  oe_t = MPI_Wtime();
  if (num_self) {
    if (sparta->kokkos->gpu_aware_flag) {
      copymode = 1;
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagIrregularUnpackBufferSelf>(0,num_self),*this);
      DeviceType().fence();
      copymode = 0;
    } else { // unpack on host
      // OpenEdge perf: persistent host mirror of the send buffer, copy only
      // the bytes in use (was create_mirror_view_and_copy of the whole view
      // every call: host alloc + full D2H per step)
      bigint used_send = (bigint)num_self*nbytes;
      for (int isend = 0; isend < nsend; isend++) used_send += (bigint)num_send[isend]*nbytes;
      if (used_send > (bigint)d_sendbuf.extent(0)) used_send = d_sendbuf.extent(0);
      if (h_sendbuf.extent(0) < d_sendbuf.extent(0))
        h_sendbuf = HAT::t_char_1d(Kokkos::view_alloc("irregular:d_sendbuf:mirror",Kokkos::WithoutInitializing),d_sendbuf.extent(0));
      if (used_send > 0)
        Kokkos::deep_copy(Kokkos::subview(h_sendbuf,std::make_pair((bigint)0,used_send)),
                          Kokkos::subview(d_sendbuf,std::make_pair((bigint)0,used_send)));

      k_index_self.sync_host();

      for (int i = 0; i < num_self; i++) {
        const int m = k_index_self.view_host()[i];
        memcpy(&h_recvbuf[(bigint)i*nbytes],&h_sendbuf[(bigint)m*nbytes],nbytes);
      }
    }
  }

  oe_xt[3] += MPI_Wtime() - oe_t;

  // wait on all incoming messages

  oe_t = MPI_Wtime();
  if (nrecv)
    MPI_Waitall(nrecv,request,status);
  oe_xt[4] += MPI_Wtime() - oe_t; oe_t = MPI_Wtime();

  if (!sparta->kokkos->gpu_aware_flag)
    if (nrecv || num_self)
    {
      // OpenEdge perf: H2D of the received bytes only (buffers are grow-only)
      bigint used_recv = (bigint)num_self*nbytes;
      for (int irecv = 0; irecv < nrecv; irecv++) used_recv += (bigint)num_recv[irecv]*nbytes;
      if (used_recv > (bigint)d_recvbuf.extent(0)) used_recv = d_recvbuf.extent(0);
      if (used_recv > 0)
        Kokkos::deep_copy(Kokkos::subview(d_recvbuf,std::make_pair((bigint)0,used_recv)),
                          Kokkos::subview(h_recvbuf,std::make_pair((bigint)0,used_recv)));
  }
  oe_xt[5] += MPI_Wtime() - oe_t;
}

KOKKOS_INLINE_FUNCTION
void IrregularKokkos::operator()(TagIrregularPackBuffer, const int &i) const {
  const int m = d_index_send[offset_send + i];
  memcpy(&d_buf[(bigint)i*nbytes],&d_sendbuf[(bigint)m*nbytes],nbytes);
}

KOKKOS_INLINE_FUNCTION
void IrregularKokkos::operator()(TagIrregularUnpackBufferSelf, const int &i) const {
  const int m = d_index_self[i];
  memcpy(&d_recvbuf[(bigint)i*nbytes],&d_sendbuf[(bigint)m*nbytes],nbytes);
}
