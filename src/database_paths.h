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

// Resolve database/surface/<proj>_on_<target>.h5. If `proj` already looks
// like a literal path, it is returned unchanged (target ignored). Fatal
// error via `error` if the file does not exist.
std::string resolve_surface_file(const std::string &proj,
                                 const std::string &target,
                                 Error *error);

// Resolve database/adas/ADAS_Rates_<Z>.h5 from an element name or a
// numeric Z (both accepted). Literal paths pass through unchanged. Fatal
// error via `error` if unresolved or missing.
std::string resolve_adas_file(const std::string &element_or_z, Error *error);

// Resolve database/adas/reactions/<element>.reactions from an element
// name. Literal paths (contain '/' or end in .reactions) pass through.
// Fatal error if unresolved or missing.
std::string resolve_reactions_file(const std::string &element_or_path,
                                   Error *error);

}

#endif
