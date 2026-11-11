// Standalone driver for CoreTransport1D on a make_inputs.py case file.
// Writes the densities after steps 1, nsteps/2, nsteps (same layout as the Aurora
// reference) and a per-step ledger: total content, source, outflux, parallel loss.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "core_transport_1d.h"

using namespace SPARTA_NS;

int main(int argc, char **argv)
{
  if (argc < 4) { fprintf(stderr, "usage: test_driver case.dat dens.out ledger.out\n"); return 2; }
  std::ifstream in(argv[1]);
  int ir, nion, nsteps, ievolve, isrc, in0, ipl;
  double dt, dlen;
  in >> ir >> nion >> nsteps >> dt >> dlen >> ievolve >> isrc >> in0 >> ipl;
  auto rd = [&](std::vector<double> &v, size_t n) { v.resize(n); for (size_t k = 0; k < n; k++) in >> v[k]; };
  std::vector<double> rr, n0, pl, diff, conv, s, r, src, n;
  const size_t nn = (size_t) nion * ir;
  rd(rr, ir); rd(n0, ir); rd(pl, ir); rd(diff, nn); rd(conv, nn); rd(s, nn); rd(r, nn); rd(src, nn); rd(n, nn);

  CoreTransport1D ct(ir, nion);
  ct.set_grid(rr.data());
  FILE *fd = fopen(argv[2], "w"), *fl = fopen(argv[3], "w");
  fprintf(fl, "# step content source outflux parloss\n");
  double content0 = 0.0;
  const int zlo = ievolve ? 0 : 1;              // fixed neutrals are not part of the inventory
  for (int z = zlo; z < nion; z++) content0 += ct.content(z, n.data());
  fprintf(fl, "0 %.17g 0 0 0\n", content0);
  for (int it = 1; it <= nsteps; it++) {
    if (argc > 4) ct.advance(dt, diff.data(), conv.data(), s.data(), r.data(), in0 ? n0.data() : nullptr,
            isrc ? src.data() : nullptr, ipl ? pl.data() : nullptr, dlen, ievolve != 0, n.data());
    else ct.step(dt, diff.data(), conv.data(), s.data(), r.data(), in0 ? n0.data() : nullptr,
            isrc ? src.data() : nullptr, ipl ? pl.data() : nullptr, dlen, ievolve != 0, n.data());
    if (argc > 4) {
      for (double v : n) if (!std::isfinite(v) || v < 0.0) return 3;
      for (int z = 0; z < nion; ++z)
        if (!std::isfinite(ct.outflux(z)) || ct.outflux(z) < 0.0) return 4;
    }
    double content = 0.0, source = 0.0, out = 0.0, loss = 0.0;
    for (int z = zlo; z < nion; z++) {
      content += ct.content(z, n.data()); out += ct.outflux(z); loss += ct.parloss(z);
      if (isrc) for (int i = 0; i < ir; i++) source += src[(size_t) z*ir + i] * ct.cell_volume(i);
    }
    if (in0) for (int i = 0; i < ir; i++)
      source += ievolve ? n0[i] * ct.cell_volume(i) : n0[i] * s[i] * ct.cell_volume(i);
    fprintf(fl, "%d %.17g %.17g %.17g %.17g\n", it, content, source, out, loss);
    if (it == 1 || it == nsteps/2 || it == nsteps) {
      fprintf(fd, "# step %d\n", it);
      for (int z = 0; z < nion; z++) {
        for (int i = 0; i < ir; i++) fprintf(fd, " %.16e", n[(size_t) z*ir + i]);
        fprintf(fd, "\n");
      }
    }
  }
  fclose(fd); fclose(fl);
  return 0;
}
