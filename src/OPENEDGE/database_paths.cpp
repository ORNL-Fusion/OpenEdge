/* ----------------------------------------------------------------------
   OpenEdge database path resolver — implementation.
------------------------------------------------------------------------- */

#include "database_paths.h"
#include "error.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cstdio>
#include <sys/stat.h>
#include <unistd.h>
#include <hdf5.h>
#include <unordered_map>

#ifndef OPENEDGE_DATABASE_DIR
#define OPENEDGE_DATABASE_DIR ""
#endif

namespace SPARTA_NS {

static bool file_exists(const std::string &p)
{
  struct stat st;
  return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::string openedge_database_dir()
{
  if (const char *env = std::getenv("OPENEDGE_ROOT")) {
    std::string r(env);
    if (!r.empty() && r.back() == '/') r.pop_back();
    return r + "/database";
  }
  const std::string compiled = OPENEDGE_DATABASE_DIR;
  if (!compiled.empty()) return compiled;
  return "database";
}

std::string openedge_database_source()
{
  if (std::getenv("OPENEDGE_ROOT")) return "OPENEDGE_ROOT";
  if (std::string(OPENEDGE_DATABASE_DIR).size()) return "compiled default";
  return "working directory";
}

static std::string working_dir()
{
  char buf[4096];
  return getcwd(buf, sizeof(buf)) ? std::string(buf) : std::string("?");
}

std::string path_report(const std::string &path)
{
  if (path.empty() || path[0] == '/') return path;
  const std::string cwd = working_dir();
  return path + " (resolved to " + cwd + "/" + path +
         "; working directory " + cwd + ")";
}

std::string file_problem(const std::string &path)
{
  struct stat st;
  if (stat(path.c_str(), &st) != 0) {
    if (errno == ENOENT || errno == ENOTDIR) return "not found";
    if (errno == EACCES) return "cannot be read (permission denied on its directory)";
    return std::string("cannot be read (") + strerror(errno) + ")";
  }
  if (!S_ISREG(st.st_mode)) return "is not a file";
  if (access(path.c_str(), R_OK) != 0) return "cannot be read (permission denied)";
  return "";
}

// HDF5's own test: the signature may follow a user block
bool file_is_hdf5(const std::string &path)
{
  htri_t ok = 0;
  H5E_BEGIN_TRY { ok = H5Fis_hdf5(path.c_str()); } H5E_END_TRY;
  return ok > 0;
}

std::string processes_file_error()
{
  const std::string p = openedge_database_dir() + "/processes.h5";
  const std::string where = p + " (database directory from " +
                            openedge_database_source() + ")";
  const std::string why = file_problem(p);
  if (why == "not found")
    return "OpenEdge database file not found: " + where +
           ". Set OPENEDGE_ROOT to the OpenEdge directory that holds "
           "database/processes.h5";
  if (why == "is not a file")
    return "OpenEdge database path is not a file: " + where +
           ". Put database/processes.h5 there, or set OPENEDGE_ROOT to the "
           "OpenEdge directory that holds it";
  if (!why.empty())
    return "OpenEdge database file " + why + ": " + where +
           ". Make it readable by this account";
  if (!file_is_hdf5(p))
    return "OpenEdge database file is not a valid HDF5 file: " + where +
           ". Restore database/processes.h5 from the repository";
  return "";
}

// Isotopes of hydrogen map to Z=1. Common fusion-relevant elements.
static const std::unordered_map<std::string,int> &element_table()
{
  static const std::unordered_map<std::string,int> tbl = {
    {"H",1}, {"D",1}, {"T",1},
    {"He",2}, {"Li",3}, {"Be",4}, {"B",5}, {"C",6}, {"N",7}, {"O",8},
    {"F",9}, {"Ne",10}, {"Na",11}, {"Mg",12}, {"Al",13}, {"Si",14},
    {"P",15}, {"S",16}, {"Cl",17}, {"Ar",18}, {"K",19}, {"Ca",20},
    {"Ti",22}, {"V",23}, {"Cr",24}, {"Mn",25}, {"Fe",26}, {"Co",27},
    {"Ni",28}, {"Cu",29}, {"Zn",30}, {"Ga",31},
    {"Kr",36}, {"Mo",42}, {"Xe",54},
    {"Ta",73}, {"W",74},
  };
  return tbl;
}

int element_to_z(const std::string &elem)
{
  const auto &tbl = element_table();
  auto it = tbl.find(elem);
  return (it == tbl.end()) ? -1 : it->second;
}

std::string element_of_ion_name(const std::string &ion_name)
{
  size_t n = ion_name.size();
  while (n > 0) {
    char c = ion_name[n-1];
    if (c == '+' || c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
      --n;
    } else break;
  }
  return ion_name.substr(0, n);
}

bool path_looks_literal(const std::string &s)
{
  if (s.empty()) return false;
  if (s.find('/') != std::string::npos) return true;
  if (s[0] == '.') return true;
  if (s.size() >= 3 && s.compare(s.size()-3, 3, ".h5") == 0) return true;
  const std::string rx = ".reactions";
  if (s.size() >= rx.size() &&
      s.compare(s.size() - rx.size(), rx.size(), rx) == 0) return true;
  return false;
}

std::string resolve_reactions_file(const std::string &spec, Error *error)
{
  std::string p;
  if (path_looks_literal(spec)) {
    p = spec;
  } else {
    if (spec.empty())
      error->all(FLERR,
                 "resolve_reactions_file: empty element / path");
    // Legacy text-file fallback when processes.h5 catalog is absent.
    // Runtime consumer (fix volume/chem/adas) prefers
    // /volume/reactions/<elem>/catalog inside processes.h5 first; only
    // hits this resolver when that lookup fails.
    p = openedge_database_dir() + "/ingest/reactions/" + spec + ".reactions";
  }
  if (!file_exists(p)) {
    // an element name falls back to this text file only when the catalogue
    // in processes.h5 could not be used: say which of the two failed
    std::string msg = "Reactions file " + file_problem(p) + ": " + path_report(p);
    if (!path_looks_literal(spec)) {
      const std::string db = processes_file_error();
      if (!db.empty()) msg = db;
      else msg = "No reactions for element '" + spec + "': no catalogue "
                 "/volume/reactions/<element>/catalog in " + openedge_database_dir() +
                 "/processes.h5 and no file " + p + ". Use an element the "
                 "database holds, or give the path of a .reactions file";
    }
    error->all(FLERR, msg.c_str());
  }
  return p;
}

// "" only when the path does not exist. Anything else there (a file that is
// unreadable or not HDF5, a directory, a path whose directory denies access)
// is returned, so that opening it stops the run instead of selecting a
// caller's fallback for a missing database
std::string resolve_processes_file()
{
  std::string p = openedge_database_dir() + "/processes.h5";
  struct stat st;
  if (stat(p.c_str(), &st) != 0 && (errno == ENOENT || errno == ENOTDIR))
    return std::string();
  return p;
}

std::string resolve_iead_file(const std::string &tag)
{
  // "light" = iead_database.h5 (D-scaled, m_proj = 2Z amu); any other tag is
  // a per-element real-mass table iead_database_<tag>.h5 (W, O, Ne, ...)
  std::string filename = "iead_database.h5";
  if (tag != "light") filename = "iead_database_" + tag + ".h5";
  std::string p = openedge_database_dir() + "/iead/" + filename;
  return file_exists(p) ? p : std::string();
}

}
