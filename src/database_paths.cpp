/* ----------------------------------------------------------------------
   OpenEdge database path resolver — implementation.
------------------------------------------------------------------------- */

#include "database_paths.h"
#include "error.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
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

std::string resolve_surface_file(const std::string &proj,
                                 const std::string &target,
                                 Error *error)
{
  std::string p;
  if (path_looks_literal(proj)) {
    p = proj;
  } else {
    if (target.empty())
      error->all(FLERR,
                 "resolve_surface_file: target element required when proj "
                 "is a short name");
    p = openedge_database_dir() + "/surface/" + proj + "_on_" + target + ".h5";
  }
  if (!file_exists(p)) {
    std::string msg = "Surface data file not found: " + p +
                      " (OPENEDGE_ROOT or OPENEDGE_DATABASE_DIR)";
    error->all(FLERR, msg.c_str());
  }
  return p;
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
    p = openedge_database_dir() + "/adas/reactions/" + spec + ".reactions";
  }
  if (!file_exists(p)) {
    std::string msg = "Reactions file not found: " + p;
    error->all(FLERR, msg.c_str());
  }
  return p;
}

std::string resolve_adas_file(const std::string &element_or_z, Error *error)
{
  if (path_looks_literal(element_or_z)) {
    if (!file_exists(element_or_z)) {
      std::string msg = "ADAS rate file not found: " + element_or_z;
      error->all(FLERR, msg.c_str());
    }
    return element_or_z;
  }

  int z = -1;
  // numeric?
  if (!element_or_z.empty() &&
      std::all_of(element_or_z.begin(), element_or_z.end(),
                  [](unsigned char c){ return std::isdigit(c); })) {
    z = std::atoi(element_or_z.c_str());
  } else {
    z = element_to_z(element_or_z);
    if (z < 0) {
      std::string msg = "Unknown element '" + element_or_z +
                        "' for ADAS file resolution";
      error->all(FLERR, msg.c_str());
    }
  }
  std::string p = openedge_database_dir() + "/adas/ADAS_Rates_" +
                  std::to_string(z) + ".h5";
  if (!file_exists(p)) {
    std::string msg = "ADAS rate file not found: " + p;
    error->all(FLERR, msg.c_str());
  }
  return p;
}

std::string resolve_processes_file()
{
  std::string p = openedge_database_dir() + "/processes.h5";
  return file_exists(p) ? p : std::string();
}

}
