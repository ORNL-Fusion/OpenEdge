/* ----------------------------------------------------------------------
   OpenEdge — Neutral and charge-state transport with plasma-wall interactions
   Built on SPARTA (sparta.github.io; Plimpton et al., Sandia National Labs).

   Author: Abdourahmane Diaw <diawa@ornl.gov>
           Oak Ridge National Laboratory
   https://github.com/ORNL-Fusion/OpenEdge

   Distributed under the GNU General Public License, same as SPARTA.
   See the LICENSE file at the top of the repository.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   ProcessLibrary: consolidated-data loader for database/processes.h5.

   Thin wrapper that maps the canonical /volume/ + /surface/ schema
   (see docs/database_schema.md) to in-memory tables consumers can
   interpolate.  Rank-0 reads from HDF5, then MPI_Bcast to all ranks --
   same pattern used throughout the OpenEdge package.  Tables are cached
   per (class, element) on first access; repeated load_rate() calls are
   O(1) lookups.

   Currently implemented:
     - load_rate(class, elem, ...)              /volume/rates/<cls>/<elem>/
                                                /volume/radiation/<cls>/<elem>/
                                                (cls = "scd", "acd", "ccd",
                                                "plt", "prb")
     - load_ionization_potential(elem, ip)      /volume/thresholds/ionization/<elem>/

   Stubs for later (return false when called; consumer falls back):
     - load_pec(elem, pec_id, line, ...)        /volume/pec/
     - load_surface_reflection(pair, ...)       /surface/reflection/
     - load_surface_sputter(pair, ...)          /surface/sputter/
------------------------------------------------------------------------- */

#ifndef SPARTA_OPENEDGE_PROCESS_LIBRARY_H
#define SPARTA_OPENEDGE_PROCESS_LIBRARY_H

#include <map>
#include <string>
#include <vector>

#include <mpi.h>

namespace SPARTA_NS {

class Error;

class ProcessLibrary {
 public:
  ProcessLibrary() = default;

  // Open processes.h5.  `path` is the absolute path (typically from
  // resolve_processes_file()); pass an empty string to leave the
  // library closed (is_open() returns false, all load_*() calls return
  // false).  `comm` is used for collective broadcasts.  Must be called
  // on every rank.  Fatal via `error` if the file is present but fails
  // to open (rank 0 detects, broadcasts the failure).
  void open(const std::string &path, MPI_Comm comm, Error *error);

  bool is_open() const { return opened_; }
  const std::string &path() const { return path_; }

  // Load an ADAS-class rate or radiation table for one element.
  // cls:   "scd" | "acd" | "ccd"  -> /volume/rates/<cls>/<elem>/
  //        "plt" | "prb"          -> /volume/radiation/<cls>/<elem>/
  // elem:  canonical lowercase symbol ("h", "he", "c", "w", ...).
  // Fills `coef` as a flat (nQ * nT * nD) vector of log10 values, and
  // the axis vectors `logT` / `logN` (units: log10 eV and log10 cm^-3
  // respectively, matching ADAS adf11 convention).
  // Returns true on success, false if the table is not present (consumer
  // should fall back to a legacy per-element file).
  bool load_rate(const std::string &cls, const std::string &elem,
                 std::vector<double> &coef,
                 std::vector<double> &logT,
                 std::vector<double> &logN,
                 int &nQ, int &nT, int &nD);

  // Load the ionization-potential vector (Z entries, in eV) for an
  // element.  Returns true on success, false if not present.
  bool load_ionization_potential(const std::string &elem,
                                 std::vector<double> &ip_eV);

 private:
  bool         opened_ = false;
  std::string  path_;
  MPI_Comm     comm_ = MPI_COMM_NULL;
  int          me_   = 0;

  // Cache loaded tables so repeated load_rate() calls don't hit HDF5
  // every time.  Key = "<cls>/<elem>".
  struct RateCacheEntry {
    bool                 present = false;
    std::vector<double>  coef;
    std::vector<double>  logT, logN;
    int                  nQ = 0, nT = 0, nD = 0;
  };
  std::map<std::string, RateCacheEntry>           rate_cache_;
  std::map<std::string, std::vector<double>>      ip_cache_;

  // Shared helpers: read + broadcast a 3D rate table or 1D vector.
  // Rank 0 opens HDF5 paths; all ranks participate in the broadcast.
  bool fetch_rate(const std::string &group_path,
                  RateCacheEntry &out, Error *error);
  bool fetch_vector(const std::string &dataset_path,
                    std::vector<double> &out, Error *error);
};

}  // namespace SPARTA_NS

#endif
