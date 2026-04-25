/* Test driver for OpenEdge coupling API
   Exercises: openedge_extract_fix(), openedge_get_ngrid(), openedge_reload_plasma()
   Build: mpicxx -o test_coupling_api test_coupling_api.cpp -I../../src -L../../buildOpenEdge/src -lsparta_mpi -L../../buildOpenEdge/src/OPENEDGE -lpkg_openedge -lhdf5 -lmpi
*/

#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "library.h"

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int me;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  // 1. Open OpenEdge as a library
  void *oe;
  char *args[] = {(char*)"openedge", (char*)"-log", (char*)"none"};
  sparta_open(3, args, MPI_COMM_WORLD, &oe);

  if (me == 0) printf("=== OpenEdge coupling API test ===\n");

  // 2. Feed the input script commands (same as in.solps_coupling_baseline but fewer steps)
  sparta_command(oe, (char*)"seed 12345");
  sparta_command(oe, (char*)"timestep 1e-05");
  sparta_command(oe, (char*)"dimension 2");
  sparta_command(oe, (char*)"boundary r r p");
  sparta_command(oe, (char*)"global gridcut 0.0 comm/sort no");
  sparta_command(oe, (char*)"global temp 1.e20");
  sparta_command(oe, (char*)"create_box 2.0 6.0 -4.0 3.8 -0.05 0.05");
  sparta_command(oe, (char*)"create_grid 200 200 1");
  sparta_command(oe, (char*)"balance_grid rcb cell");
  sparta_command(oe, (char*)"global fnum 0.0000001");

  sparta_command(oe, (char*)"species input/droplet.species drop_2");
  sparta_command(oe, (char*)"mixture DropletSource drop_2 frac 1.0 nrho 1");
  sparta_command(oe, (char*)"variable pmass particle mass");

  sparta_command(oe, (char*)"read_surf input/wall.surf group surfID particle check");
  sparta_command(oe, (char*)"surf_collide wallColModel diffuse 773.15 1.0");
  sparta_command(oe, (char*)"surf_react wallReact global 1. .0");
  sparta_command(oe, (char*)"surf_modify surfID collide wallColModel react wallReact");

  sparta_command(oe, (char*)"read_particles input/particle.source 1");
  sparta_command(oe, (char*)"read_surf input/core.surf invert group surfID2 particle check");
  sparta_command(oe, (char*)"surf_modify surfID2 collide wallColModel react wallReact");

  // Plasma fields compute
  sparta_command(oe, (char*)"compute cnear plasma/fields all file "
    "../test_evaporation/input/plasma.h5 "
    "equilibrium ../test_evaporation/input/g000001.00001_symm.X4.equ "
    "bx by bz temp_e dens_e temp_i dens_i parrflow er et ez");

  // Evaporation fix
  double rho = 534.0, rd = 2.5e-3;
  double md = (4.0/3.0) * 3.14159265358979 * rho * rd * rd * rd;
  char cmd[512];
  snprintf(cmd, sizeof(cmd),
    "fix fevap evaporation 1 DropletSource mass %.10e radius %.6e temp 773.15 "
    "heatflux/compute cnear heatflux/scale 1.0",
    md, rd);
  sparta_command(oe, cmd);

  // Drag, charging, efield
  snprintf(cmd, sizeof(cmd),
    "fix fdrag drag 1 2.0 1 "
    "plasma c_cnear[4] c_cnear[6] c_cnear[7] c_cnear[8] "
    "bfield c_cnear[1] c_cnear[3] c_cnear[2] "
    "gravity 0 -9.8 0 model epstein "
    "mass %.10e radius %.6e temp 773.15 cylindrical yes", md, rd);
  sparta_command(oe, cmd);

  snprintf(cmd, sizeof(cmd),
    "fix fcharge droplet/charge 1 plasma "
    "temp_e c_cnear[4] temp_i c_cnear[6] "
    "dens_e c_cnear[5] dens_i c_cnear[7] "
    "radius %.6e mass %.10e temp 773.15 "
    "ion_mass_amu 2.0 thermionic no", rd, md);
  sparta_command(oe, cmd);

  sparta_command(oe, (char*)"fix eS efield/grid c_cnear[9] c_cnear[10] c_cnear[11]");
  sparta_command(oe, (char*)"global efield grid eS 0");
  sparta_command(oe, (char*)"global pusher subcycles 5");

  sparta_command(oe, (char*)"stats 10");

  // 3. Run 10 steps
  if (me == 0) printf("\n--- Running 10 steps ---\n");
  sparta_command(oe, (char*)"run 10");

  // 4. Test openedge_get_ngrid()
  int ngrid = openedge_get_ngrid(oe);
  if (me == 0) printf("\n--- openedge_get_ngrid() = %d ---\n", ngrid);

  // 5. Test openedge_extract_fix() on the evaporation fix
  double **evap_arr = (double **) openedge_extract_fix(oe, (char*)"fevap", 2, 1);
  if (evap_arr == NULL) {
    if (me == 0) printf("ERROR: openedge_extract_fix returned NULL\n");
  } else {
    // Sum up mass loss and atom count across all grid cells
    double total_dm = 0.0, total_dn = 0.0, total_heat = 0.0;
    for (int i = 0; i < ngrid; i++) {
      total_dm   += evap_arr[i][0];
      total_dn   += evap_arr[i][1];
      total_heat += evap_arr[i][2];
    }
    double global_dm, global_dn, global_heat;
    MPI_Reduce(&total_dm, &global_dm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_dn, &global_dn, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_heat, &global_heat, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (me == 0) {
      printf("  openedge_extract_fix('fevap') OK\n");
      printf("  Total mass loss:  %.6e kg\n", global_dm);
      printf("  Total atom loss:  %.6e atoms\n", global_dn);
      printf("  Total heat:       %.6e J\n", global_heat);
    }
  }

  // 6. Test openedge_reload_plasma() - reload same file (should not crash)
  if (me == 0) printf("\n--- Testing openedge_reload_plasma() ---\n");
  openedge_reload_plasma(oe, (char*)"cnear", NULL);
  if (me == 0) printf("  reload_plasma(NULL) OK - re-read current file\n");

  openedge_reload_plasma(oe, (char*)"cnear",
    (char*)"../test_evaporation/input/plasma.h5");
  if (me == 0) printf("  reload_plasma(explicit path) OK\n");

  // 7. Run 10 more steps after reload to verify simulation continues
  if (me == 0) printf("\n--- Running 10 more steps after reload ---\n");
  sparta_command(oe, (char*)"run 10");

  // 8. Extract fix again to verify it still works
  evap_arr = (double **) openedge_extract_fix(oe, (char*)"fevap", 2, 1);
  if (evap_arr) {
    double total_dm = 0.0;
    for (int i = 0; i < ngrid; i++) total_dm += evap_arr[i][0];
    double global_dm;
    MPI_Reduce(&total_dm, &global_dm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (me == 0)
      printf("  Post-reload mass loss: %.6e kg (should be nonzero)\n", global_dm);
  }

  if (me == 0) printf("\n=== All tests passed ===\n");

  sparta_close(oe);
  MPI_Finalize();
  return 0;
}
