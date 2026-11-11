! Reference driver: runs Aurora's impden1 (Linder scheme, fixed-neutral mode) on a case file
! written by scripts/make_inputs.py and writes the densities after steps 1, nsteps/2, nsteps.
! evolve = 1 cases are mapped onto Aurora's fixed-neutral mode with one padded state: the fixed
! "neutral" carries density n0 and S = 1, so Aurora's first ion is the transported neutral of the
! C++ solver with volume source n0 (Aurora's own evolveneut path double counts the old neutrals
! and drops recombined ones, so it is not used).
! Build: gfortran -O2 -o aurora_ref $AURORA/aurora/math.f90 $AURORA/aurora/impden.f90 aurora_ref.f90
program aurora_ref
  implicit none
  integer :: ir, nion, nsteps, ievolve, isrc, in0, ipl, i, nz, it, species
  real*8 :: dt, dlen
  real*8, allocatable :: rr(:), n0(:), pl(:), diff(:,:), conv(:,:), s(:,:), r(:,:), src(:,:)
  real*8, allocatable :: ra(:,:), rn(:,:), zeros(:), spt(:,:), fluxm(:), fluxd(:), ym(:), yd(:)
  real*8, allocatable :: rd(:,:)
  integer :: np
  real*8, allocatable :: dsput(:), msput(:)
  real*8 :: tsu, dsu, dsul, Nmainret, Ndivret, rcld, rcld_refl, rcld_recl, div_impl
  real*8 :: rclb, rcls, rclp, rclw, rclw_refl, rclw_recl, main_impl
  character(len=256) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(10, file=trim(fin), status='old')
  read(10,*) ir, nion, nsteps
  read(10,*) dt, dlen, ievolve, isrc, in0, ipl
  np = 0
  if (ievolve /= 0) np = 1          ! padded fixed state in front of the chain
  allocate(rr(ir), n0(ir), pl(ir), diff(ir,nion+np), conv(ir,nion+np), s(ir,nion+np), r(ir,nion+np))
  allocate(rd(ir,nion), ra(ir,nion+np), rn(ir,nion+np), zeros(ir))
  read(10,*) (rr(i), i=1,ir)
  read(10,*) (n0(i), i=1,ir)
  read(10,*) (pl(i), i=1,ir)
  read(10,*) ((diff(i,nz), i=1,ir), nz=1+np,nion+np)
  read(10,*) ((conv(i,nz), i=1,ir), nz=1+np,nion+np)
  read(10,*) ((s(i,nz), i=1,ir), nz=1+np,nion+np)
  read(10,*) ((r(i,nz), i=1,ir), nz=1+np,nion+np)
  read(10,*) ((rd(i,nz), i=1,ir), nz=1,nion)
  read(10,*) ((rn(i,nz), i=1,ir), nz=1+np,nion+np)
  close(10)
  if (isrc /= 0) stop 'per-state src not supported by Aurora'
  if (in0 == 0) n0 = 0.d0
  if (ipl == 0) pl = 0.d0
  zeros = 0.d0
  if (np == 1) then
     diff(:,1) = 0.d0; conv(:,1) = 0.d0; s(:,1) = 1.d0; r(:,1) = 0.d0; rn(:,1) = n0
  end if
  nion = nion + np

  species = 1
  allocate(spt(ir,1+species), fluxm(species), fluxd(species), ym(1+species), yd(1+species))
  allocate(dsput(1+species), msput(1+species))
  spt = 0.d0; fluxm = 0.d0; fluxd = 0.d0; ym = 0.d0; yd = 0.d0
  tsu = 0.d0; dsu = 0.d0; dsul = 0.d0; Nmainret = 0.d0; Ndivret = 0.d0

  open(11, file=trim(fout), status='replace')
  do it = 1, nsteps
     ra = rn
     call impden1(nion, ir, species, ra, rn, diff, conv, pl, n0, zeros, zeros, spt, .false., &
          s, r, 100.d0, rr, dlen, dt, &
          1.d0, 1.d0, .false., 74, 74, 0.d0, 0.d0, fluxm, fluxd, ym, yd, 0.d0, 0.d0, 0.d0, 0.d0, &
          -1.d0, 0.d0, 0.d0, tsu, dsu, dsul, 0.d0, 0.d0, 1.d0, 0.d0, 0.d0, 1.d0, &
          .false., Nmainret, Ndivret, rcld, rcld_refl, rcld_recl, div_impl, dsput, &
          rclb, rcls, rclp, rclw, rclw_refl, rclw_recl, main_impl, msput)
     if (it == 1 .or. it == nsteps/2 .or. it == nsteps) then
        write(11,'(A,I6)') '# step', it
        do nz = 1+np, nion
           write(11,'(*(ES25.16E3))') (rn(i,nz), i=1,ir)
        end do
     end if
  end do
  close(11)
end program aurora_ref
