/* ----------------------------------------------------------------------
   OpenEdge: store one named per-particle transport-force contribution.
------------------------------------------------------------------------- */

#include "fix_store_force.h"

#include <cstdio>
#include <cstring>
#include <string>

#include "error.h"
#include "input.h"
#include "modify.h"
#include "particle.h"
#include "pusher.h"
#include "update.h"

using namespace SPARTA_NS;

namespace {

FixStoreForce::Channel parse_channel(const char *name, Error *error)
{
  if (strcmp(name, "electric/plasma") == 0)
    return FixStoreForce::ELECTRIC_PLASMA;
  if (strcmp(name, "electric/sheath") == 0)
    return FixStoreForce::ELECTRIC_SHEATH;
  if (strcmp(name, "magnetic") == 0)
    return FixStoreForce::MAGNETIC;
  if (strcmp(name, "thermal/ion") == 0)
    return FixStoreForce::THERMAL_ION;
  if (strcmp(name, "thermal/electron") == 0)
    return FixStoreForce::THERMAL_ELECTRON;
  if (strcmp(name, "coulomb/background") == 0)
    return FixStoreForce::COULOMB_BACKGROUND;
  if (strcmp(name, "coulomb/binary") == 0)
    return FixStoreForce::COULOMB_BINARY;
  error->all(FLERR, "fix store/force: unknown force channel");
  return FixStoreForce::ELECTRIC_PLASMA;
}

}  // namespace

const char *FixStoreForce::channel_name(Channel channel)
{
  static const char *names[NCHANNEL] = {
    "electric/plasma",
    "electric/sheath",
    "magnetic",
    "thermal/ion",
    "thermal/electron",
    "coulomb/background",
    "coulomb/binary"
  };
  return names[static_cast<int>(channel)];
}

FixStoreForce::FixStoreForce(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg), channel_(ELECTRIC_PLASMA), id_max_(0),
  force_custom_(-1), step_custom_(-1)
{
  if (narg < 3)
    error->all(FLERR,
      "Illegal fix store/force command (need: channel [id_max ID])");
  channel_ = parse_channel(arg[2], error);

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "id_max") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix store/force id_max needs an ID");
      id_max_ = input->bnumeric(FLERR, arg[iarg + 1]);
      if (id_max_ < 0)
        error->all(FLERR, "fix store/force id_max must be non-negative");
      iarg += 2;
    } else {
      error->all(FLERR, "fix store/force: unknown keyword");
    }
  }

  // Particle custom arrays migrate, reorder and restart with their particle.
  // The fix aliases the force array through array_particle so dump particle
  // can use the familiar f_ID[1], f_ID[2], f_ID[3] interface.
  const int DOUBLE = 1;
  const std::string force_name = std::string("store_force_") + id;
  const std::string step_name = std::string("store_force_step_") + id;
  force_custom_ = particle->find_custom(
    const_cast<char *>(force_name.c_str()));
  if (force_custom_ < 0)
    force_custom_ = particle->add_custom(
      const_cast<char *>(force_name.c_str()), DOUBLE, 3);
  else if (particle->etype[force_custom_] != DOUBLE ||
           particle->esize[force_custom_] != 3)
    error->all(FLERR, "fix store/force force storage has incompatible type");

  step_custom_ = particle->find_custom(
    const_cast<char *>(step_name.c_str()));
  if (step_custom_ < 0)
    step_custom_ = particle->add_custom(
      const_cast<char *>(step_name.c_str()), DOUBLE, 0);
  else if (particle->etype[step_custom_] != DOUBLE ||
           particle->esize[step_custom_] != 0)
    error->all(FLERR, "fix store/force step storage has incompatible type");

  per_particle_flag = 1;
  size_per_particle_cols = 3;
  per_particle_freq = 1;
  time_depend = 1;
  vector_particle = nullptr;
  refresh_output();
}

FixStoreForce::~FixStoreForce()
{
  if (copy || copymode) return;
  array_particle = nullptr;
  if (force_custom_ >= 0) particle->remove_custom(force_custom_);
  if (step_custom_ >= 0) particle->remove_custom(step_custom_);
}

int FixStoreForce::setmask()
{
  int mask = 0;
  mask |= START_OF_STEP;
  mask |= END_OF_STEP;
  return mask;
}

void FixStoreForce::init()
{
  int same_channel = 0;
  for (int i = 0; i < modify->nfix; ++i) {
    auto *store = dynamic_cast<FixStoreForce *>(modify->fix[i]);
    if (store && store->channel() == channel_) ++same_channel;
  }
  if (same_channel != 1) {
    char message[256];
    std::snprintf(message, sizeof(message),
      "Only one fix store/force may store channel %s",
      channel_name(channel_));
    error->all(FLERR, message);
  }

  if ((channel_ == ELECTRIC_PLASMA || channel_ == ELECTRIC_SHEATH ||
       channel_ == MAGNETIC) &&
      (!update->pusher || update->pusher->pusher_mode != Pusher::PUSHER_BORIS))
    error->all(FLERR,
      "Pusher force channels in fix store/force currently require mode boris");
  if (sparta->kokkos)
    error->all(FLERR, "fix store/force is currently available only on the CPU path");
  refresh_output();
}

bool FixStoreForce::enabled(int i) const
{
  return i >= 0 && i < particle->nlocal &&
    (id_max_ == 0 || particle->particles[i].id <= id_max_);
}

void FixStoreForce::refresh_output()
{
  array_particle =
    particle->edarray[particle->ewhich[force_custom_]];
}

void FixStoreForce::touch(int i)
{
  if (!enabled(i)) return;
  refresh_output();
  double *last_step = particle->edvec[particle->ewhich[step_custom_]];
  const double step = static_cast<double>(update->ntimestep);
  if (last_step[i] != step) {
    array_particle[i][0] = 0.0;
    array_particle[i][1] = 0.0;
    array_particle[i][2] = 0.0;
    last_step[i] = step;
  }
}

void FixStoreForce::add_impulse(int i, const double dp[3], double interval)
{
  if (!enabled(i)) return;
  if (!(interval > 0.0))
    error->all(FLERR, "fix store/force received a non-positive interval");
  touch(i);
  array_particle[i][0] += dp[0] / interval;
  array_particle[i][1] += dp[1] / interval;
  array_particle[i][2] += dp[2] / interval;
}

void FixStoreForce::start_of_step()
{
  for (int i = 0; i < particle->nlocal; ++i) touch(i);
}

void FixStoreForce::end_of_step()
{
  // Particle growth or migration can reallocate/reorder the custom array.
  // Its values migrate with particles; only the public alias needs refresh.
  refresh_output();
}

double FixStoreForce::memory_usage()
{
  // Storage is owned and reported by Particle's custom-attribute arrays.
  return 0.0;
}

FixStoreForce *SPARTA_NS::find_store_force(Modify *modify,
                                           FixStoreForce::Channel channel)
{
  if (!modify) return nullptr;
  for (int i = 0; i < modify->nfix; ++i) {
    auto *store = dynamic_cast<FixStoreForce *>(modify->fix[i]);
    if (store && store->channel() == channel) return store;
  }
  return nullptr;
}
