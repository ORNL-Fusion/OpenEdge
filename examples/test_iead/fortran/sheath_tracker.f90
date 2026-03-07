!======================================================================
! sheath_tracker.f90 — Fortran+OpenMP Ta mixture Chodura sheath tracker
!
! Tracks Ta2+/Ta3+/Ta4+ ions through a magnetised sheath with
! oblique B-field.  Boris pusher, OpenMP parallel over particles.
! Outputs CSV files for each tilt angle: iead_alpha<N>.csv
!
! Compile:
!   gfortran -O3 -fopenmp -o sheath_tracker sheath_tracker.f90
! Run:
!   OMP_NUM_THREADS=8 ./sheath_tracker
!======================================================================
program sheath_tracker_main
  !$ use omp_lib
  implicit none

  ! ─── Constants ─────────────────────────────────────────────
  double precision, parameter :: QE   = 1.602176634d-19
  double precision, parameter :: AMU  = 1.660539067d-27
  double precision, parameter :: EPS0 = 8.854187817d-12
  double precision, parameter :: PI   = 3.14159265358979323846d0

  ! ─── Plasma parameters ────────────────────────────────────
  double precision, parameter :: Te_eV  = 5.0d0
  double precision, parameter :: Ti_eV  = 5.0d0
  double precision, parameter :: ni0    = 1.0d19
  double precision, parameter :: Bmag   = 0.5d0
  double precision, parameter :: Ta_amu = 180.948d0

  ! ─── Species ──────────────────────────────────────────────
  integer, parameter :: NSPEC = 3
  integer    :: Zs(NSPEC)
  double precision :: fracs(NSPEC)

  ! ─── Tilt angles ──────────────────────────────────────────
  integer, parameter :: NANG = 3
  double precision :: alpha_degs(NANG)

  ! ─── Simulation parameters ────────────────────────────────
  integer, parameter :: NP_TOTAL  = 1000000
  integer, parameter :: MAX_STEPS = 5000000

  ! ─── Derived quantities ───────────────────────────────────
  double precision :: Te, Ti, lD, m_ion
  double precision :: cs_sp(NSPEC), wci_sp(NSPEC), rho_sp(NSPEC)
  double precision :: cs_min, wci_max
  ! Background D+ sets the sheath / domain scale
  double precision :: m_D, cs_D, wci_D, rho_D
  double precision :: Lz, dt_sim
  ! Stangeby sheath model (angle-dependent, set per angle)
  double precision :: ME_kg
  double precision :: phi_total_eV       ! total wall potential [eV]
  double precision :: phi_ds_eV          ! Debye-sheath drop [eV]
  double precision :: phi_cs_eV          ! Chodura/MPS drop  [eV]
  double precision :: lmps               ! MPS scale length = rho_D [m]
  double precision :: two_lD             ! 2*lambda_D [m]

  ! ─── Per-particle output buffers (heap-allocated) ─────────
  double precision, allocatable :: p_energy(:), p_angle(:)
  integer, allocatable :: p_hit(:)

  ! ─── Locals ───────────────────────────────────────────────
  integer :: iang, ispec, ip, Np_sp, n_hit, n_lost, iunit
  double precision :: alpha_rad, sin_a, Bvec(3)
  double precision :: t_wall_start, t_wall_end
  double precision :: mean_en, mean_ang
  character(len=256) :: fname

  ! ═══════════════════════════════════════════════════════════
  !  Initialise
  ! ═══════════════════════════════════════════════════════════
  Zs         = (/ 2, 3, 4 /)
  fracs      = (/ 0.08d0, 0.62d0, 0.30d0 /)
  alpha_degs = (/ 0.0d0, 45.0d0, 85.0d0 /)

  Te    = Te_eV * QE
  Ti    = Ti_eV * QE
  lD    = sqrt(EPS0 * Te / (ni0 * QE**2))
  m_ion = Ta_amu * AMU

  do ispec = 1, NSPEC
    cs_sp(ispec)  = sqrt(dble(Zs(ispec)) * Te / m_ion)
    wci_sp(ispec) = dble(Zs(ispec)) * QE * Bmag / m_ion
    rho_sp(ispec) = cs_sp(ispec) / wci_sp(ispec)
  end do

  ! Background D+ sets the sheath scale
  m_D   = 2.0d0 * AMU
  cs_D  = sqrt(Te / m_D)            ! D+ sound speed
  wci_D = QE * Bmag / m_D           ! D+ cyclotron freq
  rho_D = cs_D / wci_D              ! D+ sound gyroradius

  cs_min  = minval(cs_sp)
  wci_max = maxval(wci_sp)
  ME_kg   = 9.1093837015d-31

  ! Domain: 15 rho_D (captures MPS) + 20 lambda_D (captures Debye sheath)
  lmps   = rho_D                     ! MPS scale length
  two_lD = 2.0d0 * lD                ! Debye sheath scale length
  Lz     = 25.0d0 * rho_D + 40.0d0 * lD

  ! Total wall potential (Stangeby floating-wall formula)
  ! phi_float = 0.5 * ln[ mD / (2 pi me) / (1 + Ti/Te) ] * Te
  phi_total_eV = 0.5d0 * log(m_D / (2.0d0*PI*ME_kg) / (1.0d0 + Ti_eV/Te_eV)) * Te_eV
  ! phi_ds and phi_cs are set per angle in the main loop

  dt_sim  = 0.1d0 * min(2.0d0 * PI / wci_max, lD / cs_min)

  ! ═══════════════════════════════════════════════════════════
  !  Banner
  ! ═══════════════════════════════════════════════════════════
  write(*,'(A)') '======================================================'
  write(*,'(A)') '  Ta Chodura sheath tracker   (Fortran + OpenMP)'
  write(*,'(A)') '======================================================'
  write(*,'(A,F6.1,A)')    '  Te = Ti  = ', Te_eV, ' eV'
  write(*,'(A,ES9.2,A)')   '  ni       = ', ni0, ' m^-3'
  write(*,'(A,F5.2,A)')    '  |B|      = ', Bmag, ' T'
  write(*,'(A,ES10.3,A)')  '  lam_D    = ', lD, ' m'
  write(*,'(A,F7.3,A)')    '  rho_D+   = ', rho_D*1d3, ' mm  (D+ sets domain)'
  write(*,'(A,ES10.3,A)')  '  Lz       = ', Lz, ' m'
  write(*,'(A,F7.2,A)')    '  phi_tot  = ', phi_total_eV, ' eV  (Stangeby floating)'
  write(*,'(A,ES10.3,A)')  '  dt       = ', dt_sim, ' s'
  write(*,'(A,I8)')         '  Np_total = ', NP_TOTAL
  write(*,'(A,I8)')         '  max_step = ', MAX_STEPS
  !$ write(*,'(A,I4)')      '  threads  = ', omp_get_max_threads()
  write(*,'(A)') '------------------------------------------------------'
  do ispec = 1, NSPEC
    write(*,'(A,I1,A,F5.2,A,F8.1,A,ES9.2,A,F7.3,A)') &
      '  Ta', Zs(ispec), '+  frac=', fracs(ispec), &
      '  cs=', cs_sp(ispec), ' m/s  wci=', wci_sp(ispec), &
      ' rad/s  rho_s=', rho_sp(ispec)*1d3, ' mm'
  end do
  write(*,'(A)') '------------------------------------------------------'

  ! ═══════════════════════════════════════════════════════════
  !  Main loop: angles
  ! ═══════════════════════════════════════════════════════════
  do iang = 1, NANG
    alpha_rad = alpha_degs(iang) * PI / 180.0d0
    Bvec(1) = Bmag * sin(alpha_rad)
    Bvec(2) = 0.0d0
    Bvec(3) = Bmag * cos(alpha_rad)

    ! Stangeby angle-dependent DS / CS split
    ! sin(alpha_grazing) = cos(alpha_from_normal)
    sin_a = max(abs(cos(alpha_rad)), 1.0d-12)
    phi_cs_eV = min(phi_total_eV, max(0.0d0, -Te_eV * log(sin_a)))
    phi_ds_eV = max(phi_total_eV - phi_cs_eV, 0.0d0)

    write(*,'(/,A,F5.1,A)') '  >>> alpha = ', alpha_degs(iang), ' deg'
    write(*,'(A,F7.2,A,F7.2,A,F7.2,A)') &
      '      phi_total=', phi_total_eV, ' eV  phi_DS=', phi_ds_eV, &
      ' eV  phi_CS=', phi_cs_eV, ' eV'

    ! Open CSV
    write(fname, '(A,I0,A)') 'iead_alpha', nint(alpha_degs(iang)), '.csv'
    iunit = 20 + iang
    open(unit=iunit, file=trim(fname), status='replace')
    write(iunit, '(A)') 'species_Z,energy_eV,angle_deg'

    t_wall_start = 0.0d0
    !$ t_wall_start = omp_get_wtime()

    ! ── Loop over species ──
    do ispec = 1, NSPEC
      Np_sp = max(100, nint(NP_TOTAL * fracs(ispec)))

      allocate(p_energy(Np_sp), p_angle(Np_sp), p_hit(Np_sp))
      p_hit    = 0
      p_energy = 0.0d0
      p_angle  = 0.0d0

      !$OMP PARALLEL DO PRIVATE(ip) SCHEDULE(DYNAMIC, 32)
      do ip = 1, Np_sp
        call track_particle(ip, ispec, iang, Zs(ispec), m_ion, &
                            cs_sp(ispec), rho_sp(ispec), Bvec,  &
                            p_energy(ip), p_angle(ip), p_hit(ip))
      end do
      !$OMP END PARALLEL DO

      ! Collect and write hits
      n_hit  = 0
      n_lost = 0
      mean_en  = 0.0d0
      mean_ang = 0.0d0
      do ip = 1, Np_sp
        if (p_hit(ip) == 1) then
          n_hit = n_hit + 1
          mean_en  = mean_en  + p_energy(ip)
          mean_ang = mean_ang + p_angle(ip)
          write(iunit, '(I1,A,F12.4,A,F12.4)') &
            Zs(ispec), ',', p_energy(ip), ',', p_angle(ip)
        else
          n_lost = n_lost + 1
        end if
      end do
      if (n_hit > 0) then
        mean_en  = mean_en  / dble(n_hit)
        mean_ang = mean_ang / dble(n_hit)
      end if

      write(*,'(A,I1,A,I9,A,I9,A,I7,A,F7.1,A,F6.1,A)') &
        '    Ta', Zs(ispec), '+  Np=', Np_sp, &
        '  hits=', n_hit, '  lost=', n_lost, &
        '  <E>=', mean_en, ' eV  <th>=', mean_ang, ' deg'

      deallocate(p_energy, p_angle, p_hit)
    end do  ! ispec

    t_wall_end = 0.0d0
    !$ t_wall_end = omp_get_wtime()
    !$ write(*,'(A,F8.2,A)') '    Wall time: ', t_wall_end - t_wall_start, ' s'

    close(iunit)
    write(*,'(A,A)') '    Saved -> ', trim(fname)
  end do  ! iang

  write(*,'(/,A)') '======================================================'
  write(*,'(A)')   '  Done.  Run:  python3 plot_iead.py'
  write(*,'(A)')   '======================================================'

contains

  ! ════════════════════════════════════════════════════════════
  !  Sheath potential phi(z) [V]   (Stangeby double-exponential)
  !
  !  phi(z) = -(phi_ds * exp(-d/2lD) + phi_cs * exp(-d/rho_D))
  !  where d = Lz - z is distance from wall.
  !  At wall (z=Lz, d=0):  phi = -(phi_ds + phi_cs)  (most negative)
  !  At entrance (z=0):    phi -> 0
  ! ════════════════════════════════════════════════════════════
  pure function sheath_phi(z) result(phi)
    double precision, intent(in) :: z
    double precision :: phi, d
    d = max(Lz - z, 0.0d0)
    phi = -(phi_ds_eV * exp(-d / two_lD) + phi_cs_eV * exp(-d / lmps))
  end function sheath_phi

  ! ════════════════════════════════════════════════════════════
  !  Sheath Ez = -dphi/dz [V/m]  (analytic derivative)
  !
  !  Ez = phi_ds/(2lD) * exp(-d/2lD) + phi_cs/rho_D * exp(-d/rho_D)
  !  Always >= 0 (pushes positive ions toward wall).
  ! ════════════════════════════════════════════════════════════
  pure function sheath_Ez(z) result(Ez)
    double precision, intent(in) :: z
    double precision :: Ez, d
    d = max(Lz - z, 0.0d0)
    Ez = phi_ds_eV / two_lD * exp(-d / two_lD) &
       + phi_cs_eV / lmps   * exp(-d / lmps)
  end function sheath_Ez

  ! ════════════════════════════════════════════════════════════
  !  xorshift64 RNG  (thread-safe, per-particle state)
  ! ════════════════════════════════════════════════════════════
  subroutine xor_uniform(state, val)
    integer(8), intent(inout) :: state
    double precision, intent(out) :: val
    if (state == 0_8) state = 1_8
    state = ieor(state, ishft(state, 13))
    state = ieor(state, ishft(state, -7))
    state = ieor(state, ishft(state, 17))
    val = abs(dble(state)) / dble(huge(state))
    val = max(val, 1.0d-15)   ! avoid exact zero for log()
  end subroutine xor_uniform

  subroutine xor_normal(state, val)
    integer(8), intent(inout) :: state
    double precision, intent(out) :: val
    double precision :: u1, u2
    call xor_uniform(state, u1)
    call xor_uniform(state, u2)
    val = sqrt(-2.0d0 * log(u1)) * cos(2.0d0 * PI * u2)
  end subroutine xor_normal

  ! ════════════════════════════════════════════════════════════
  !  Track one particle until wall hit or loss
  ! ════════════════════════════════════════════════════════════
  subroutine track_particle(ip, ispec, iang, Zion, mass, cs_loc, &
                            rho_loc, Bv, out_en, out_ang, out_flag)
    integer, intent(in)  :: ip, ispec, iang, Zion
    double precision, intent(in)  :: mass, cs_loc, rho_loc, Bv(3)
    double precision, intent(out) :: out_en, out_ang
    integer, intent(out) :: out_flag

    double precision :: pos(3), vel(3)
    double precision :: q, hqm, vth, bnorm, bhat(3)
    double precision :: Ez_val, vm(3), tv(3), sv(3), t2, vp(3), vr(3)
    double precision :: v2, vt, z_lost
    integer(8) :: rng
    double precision :: rn
    integer :: step

    q    = dble(Zion) * QE
    hqm  = 0.5d0 * dt_sim * q / mass
    vth  = sqrt(Ti / mass)
    z_lost = -5.0d0 * rho_loc

    bnorm  = sqrt(Bv(1)**2 + Bv(2)**2 + Bv(3)**2)
    bhat   = Bv / bnorm

    ! Seed RNG uniquely per (particle, species, angle)
    rng = int(ip, 8) * 6364136223846793005_8 &
        + int(ispec, 8) * 1442695040888963407_8 &
        + int(iang, 8)  * 2862933555777941757_8
    if (rng == 0_8) rng = 1_8

    ! ─── Initialise position & velocity ───
    pos = 0.0d0

    call xor_normal(rng, rn); vel(1) = cs_loc * bhat(1) + vth * rn
    call xor_normal(rng, rn); vel(2) = cs_loc * bhat(2) + vth * rn
    ! Full thermal in z; reject-resample to ensure vz > 0 (entering sheath)
    do
      call xor_normal(rng, rn)
      vel(3) = cs_loc * bhat(3) + vth * rn
      if (vel(3) > 0.0d0) exit
    end do

    out_flag = 0
    out_en   = 0.0d0
    out_ang  = 0.0d0

    ! Pre-compute rotation vectors (constant B)
    tv = hqm * Bv
    t2 = tv(1)**2 + tv(2)**2 + tv(3)**2
    sv = 2.0d0 * tv / (1.0d0 + t2)

    ! ─── Time-stepping loop ───
    do step = 1, MAX_STEPS
      ! E-field
      Ez_val = sheath_Ez(pos(3))

      ! Half E-kick
      vm(1) = vel(1)
      vm(2) = vel(2)
      vm(3) = vel(3) + hqm * Ez_val

      ! Magnetic rotation: v' = vm + vm x t
      vp(1) = vm(1) + vm(2)*tv(3) - vm(3)*tv(2)
      vp(2) = vm(2) + vm(3)*tv(1) - vm(1)*tv(3)
      vp(3) = vm(3) + vm(1)*tv(2) - vm(2)*tv(1)

      ! vr = vm + v' x s
      vr(1) = vm(1) + vp(2)*sv(3) - vp(3)*sv(2)
      vr(2) = vm(2) + vp(3)*sv(1) - vp(1)*sv(3)
      vr(3) = vm(3) + vp(1)*sv(2) - vp(2)*sv(1)

      ! Half E-kick
      vel(1) = vr(1)
      vel(2) = vr(2)
      vel(3) = vr(3) + hqm * Ez_val

      ! Position update
      pos(1) = pos(1) + dt_sim * vel(1)
      pos(2) = pos(2) + dt_sim * vel(2)
      pos(3) = pos(3) + dt_sim * vel(3)

      ! ── Wall hit ──
      if (pos(3) >= Lz) then
        out_flag = 1
        v2 = vel(1)**2 + vel(2)**2 + vel(3)**2
        out_en  = 0.5d0 * mass * v2 / QE              ! eV
        vt = sqrt(vel(1)**2 + vel(2)**2)
        out_ang = atan2(vt, abs(vel(3))) * 180.0d0 / PI
        return
      end if

      ! ── Lost upstream ──
      if (pos(3) < z_lost) return

    end do
    ! Particle still flying — flag stays 0

  end subroutine track_particle

end program sheath_tracker_main
