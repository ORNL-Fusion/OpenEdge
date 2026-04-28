/* ----------------------------------------------------------------------
   OpenEdge database path resolver.
   Maps element-name / short-name inputs to absolute HDF5 paths under
   the OpenEdge database/ tree. Lookup order:
     1. OPENEDGE_ROOT env var (appends /database)
     2. OPENEDGE_DATABASE_DIR compile-time define (set by CMake)
     3. literal "database" (cwd-relative, fallback)
------------------------------------------------------------------------- */

#ifndef SPARTA_OPENEDGE_DATABASE_PATHS_H
#define SPARTA_OPENEDGE_DATABASE_PATHS_H

#include <string>

namespace SPARTA_NS {

class Error;

// Root of the database/ tree (no trailing slash).
std::string openedge_database_dir();

// element -> atomic number. -1 on unknown element.
int element_to_z(const std::string &elem);

// Strip trailing charge state from an ion name: "D+"->"D", "O2+"->"O",
// "W5+"->"W", "D"->"D". Returns the empty string if input is empty.
std::string element_of_ion_name(const std::string &ion_name);

// True if the string looks like a filesystem path the user wrote
// explicitly (contains '/', starts with '.', or ends in '.h5').
bool path_looks_literal(const std::string &s);

// Resolve database/adas/reactions/<element>.reactions from an element
// name. Literal paths (contain '/' or end in .reactions) pass through.
// Fatal error if unresolved or missing.
std::string resolve_reactions_file(const std::string &element_or_path,
                                   Error *error);

// Resolve database/processes.h5 (the consolidated volume + surface
// process data). Returns the absolute path if the file exists, or an
// empty string if not. Non-fatal: consumers can fall back to legacy
// per-element files when processes.h5 is not shipped.
std::string resolve_processes_file();

// Resolve the IEAD lookup table for a given projectile-mass class.
// `tag = "light"` resolves database/iead/iead_database.h5 (the
// D-scaled table covering Z=1..10 elements up to Ne). `tag = "W"`
// resolves database/iead/iead_database_W.h5 (separate W table).
// Returns absolute path if present, empty string otherwise. Non-fatal:
// consumers may fall back to mean-impact yield when missing.
std::string resolve_iead_file(const std::string &tag = "light");

}

#endif
