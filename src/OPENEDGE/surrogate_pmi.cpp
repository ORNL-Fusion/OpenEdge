/* ----------------------------------------------------------------------
    OpenEdge: Lightweight NN surrogate for PMI inference
    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
      - Austin Nichols (ORNL, nicholsa@ornl.gov, 2025)
    https://github.com/ORNL-Fusion/OpenEdge

    See surrogate_pmi.h for documentation.
------------------------------------------------------------------------- */

#include "surrogate_pmi.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <H5Cpp.h>
#include <hdf5.h>

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

SurrogatePMI::SurrogatePMI()
  : loaded(false), n_input(0), n_output(4),
    n_ensemble(5), n_hidden(64), n_layers(3)
{
}

SurrogatePMI::~SurrogatePMI()
{
}

/* ---------------------------------------------------------------------- */

int SurrogatePMI::load_weights(const std::string &path)
{
  try {
    H5::H5File file(path, H5F_ACC_RDONLY);

    // Read metadata
    if (H5Lexists(file.getId(), "metadata", H5P_DEFAULT) > 0) {
      H5::Group meta = file.openGroup("metadata");
      if (H5Lexists(meta.getId(), "n_input", H5P_DEFAULT) > 0) {
        H5::DataSet ds = meta.openDataSet("n_input");
        ds.read(&n_input, H5::PredType::NATIVE_INT);
      }
      if (H5Lexists(meta.getId(), "n_output", H5P_DEFAULT) > 0) {
        H5::DataSet ds = meta.openDataSet("n_output");
        ds.read(&n_output, H5::PredType::NATIVE_INT);
      }
      if (H5Lexists(meta.getId(), "n_ensemble", H5P_DEFAULT) > 0) {
        H5::DataSet ds = meta.openDataSet("n_ensemble");
        ds.read(&n_ensemble, H5::PredType::NATIVE_INT);
      }
      if (H5Lexists(meta.getId(), "n_hidden", H5P_DEFAULT) > 0) {
        H5::DataSet ds = meta.openDataSet("n_hidden");
        ds.read(&n_hidden, H5::PredType::NATIVE_INT);
      }
      if (H5Lexists(meta.getId(), "n_layers", H5P_DEFAULT) > 0) {
        H5::DataSet ds = meta.openDataSet("n_layers");
        ds.read(&n_layers, H5::PredType::NATIVE_INT);
      }
    }

    if (n_input <= 0) {
      // Auto-detect from first layer weights shape
      std::string wpath = "ensemble/net0/layer0/weights";
      if (H5Lexists(file.getId(), wpath.c_str(), H5P_DEFAULT) > 0) {
        H5::DataSet ds = file.openDataSet(wpath);
        H5::DataSpace sp = ds.getSpace();
        hsize_t dims[2];
        sp.getSimpleExtentDims(dims);
        n_input = static_cast<int>(dims[0]);
      }
    }

    ensemble.resize(n_ensemble);

    auto read_layer = [&](const std::string &group_path) -> Layer {
      Layer lyr;
      std::string wpath = group_path + "/weights";
      std::string bpath = group_path + "/biases";

      H5::DataSet wds = file.openDataSet(wpath);
      H5::DataSpace wsp = wds.getSpace();
      int ndims = wsp.getSimpleExtentNdims();

      if (ndims == 2) {
        hsize_t dims[2];
        wsp.getSimpleExtentDims(dims);
        lyr.n_in = static_cast<int>(dims[0]);
        lyr.n_out = static_cast<int>(dims[1]);
        lyr.weights.resize(dims[0] * dims[1]);
        wds.read(lyr.weights.data(), H5::PredType::NATIVE_DOUBLE);
      } else {
        hsize_t dim;
        wsp.getSimpleExtentDims(&dim);
        lyr.n_in = 1;
        lyr.n_out = static_cast<int>(dim);
        lyr.weights.resize(dim);
        wds.read(lyr.weights.data(), H5::PredType::NATIVE_DOUBLE);
      }

      H5::DataSet bds = file.openDataSet(bpath);
      H5::DataSpace bsp = bds.getSpace();
      hsize_t bdim;
      bsp.getSimpleExtentDims(&bdim);
      lyr.biases.resize(bdim);
      bds.read(lyr.biases.data(), H5::PredType::NATIVE_DOUBLE);

      return lyr;
    };

    for (int inet = 0; inet < n_ensemble; inet++) {
      std::string base = "ensemble/net" + std::to_string(inet);
      Network &net = ensemble[inet];
      net.hidden.resize(n_layers);

      for (int il = 0; il < n_layers; il++) {
        std::string lpath = base + "/layer" + std::to_string(il);
        net.hidden[il] = read_layer(lpath);
      }

      net.output = read_layer(base + "/output");
    }

    file.close();
    loaded = true;
    return 0;

  } catch (const H5::Exception &e) {
    return -1;
  }
}

/* ---------------------------------------------------------------------- */

std::vector<double> SurrogatePMI::forward(
    const Network &net, const std::vector<double> &input) const
{
  std::vector<double> x = input;

  for (const auto &lyr : net.hidden) {
    std::vector<double> y(lyr.n_out, 0.0);
    for (int j = 0; j < lyr.n_out; j++) {
      double sum = lyr.biases[j];
      for (int i = 0; i < lyr.n_in; i++) {
        sum += x[i] * lyr.weights[i * lyr.n_out + j];
      }
      y[j] = relu(sum);
    }
    x = y;
  }

  // Output layer (linear activation)
  std::vector<double> out(net.output.n_out, 0.0);
  for (int j = 0; j < net.output.n_out; j++) {
    double sum = net.output.biases[j];
    for (int i = 0; i < net.output.n_in; i++) {
      sum += x[i] * net.output.weights[i * net.output.n_out + j];
    }
    out[j] = sum;
  }

  return out;
}

/* ---------------------------------------------------------------------- */

SurrogateOutput SurrogatePMI::evaluate(
    double E_eV, double theta_deg,
    const std::vector<double> &composition) const
{
  SurrogateOutput result;
  result.RN = 0.0;
  result.RE = 0.0;
  result.Y = 0.0;
  result.implant_depth = 0.0;
  result.uncertainty = 1.0e30;  // max uncertainty if not loaded

  if (!loaded || ensemble.empty()) return result;

  // Build input vector: [log10(E), theta/90, composition...]
  std::vector<double> input;
  input.push_back(std::log10(std::max(E_eV, 1.0)));
  input.push_back(theta_deg / 90.0);
  for (const auto &c : composition)
    input.push_back(c);

  // Pad to n_input if needed
  while (static_cast<int>(input.size()) < n_input)
    input.push_back(0.0);

  // Evaluate ensemble
  std::vector<std::vector<double>> predictions(n_ensemble);
  for (int inet = 0; inet < n_ensemble; inet++) {
    predictions[inet] = forward(ensemble[inet], input);
  }

  // Mean prediction
  std::vector<double> mean(n_output, 0.0);
  for (int inet = 0; inet < n_ensemble; inet++) {
    for (int j = 0; j < n_output; j++) {
      mean[j] += predictions[inet][j];
    }
  }
  for (int j = 0; j < n_output; j++)
    mean[j] /= n_ensemble;

  // Standard deviation (uncertainty)
  std::vector<double> stddev(n_output, 0.0);
  for (int inet = 0; inet < n_ensemble; inet++) {
    for (int j = 0; j < n_output; j++) {
      double diff = predictions[inet][j] - mean[j];
      stddev[j] += diff * diff;
    }
  }
  for (int j = 0; j < n_output; j++)
    stddev[j] = std::sqrt(stddev[j] / n_ensemble);

  // Map outputs: [RN, RE, Y, implant_depth]
  if (n_output >= 1) result.RN = std::max(0.0, std::min(1.0, mean[0]));
  if (n_output >= 2) result.RE = std::max(0.0, std::min(1.0, mean[1]));
  if (n_output >= 3) result.Y = std::max(0.0, mean[2]);
  if (n_output >= 4) result.implant_depth = std::max(0.0, mean[3]);

  // Uncertainty: max std dev across outputs, normalized
  result.uncertainty = *std::max_element(stddev.begin(), stddev.end());

  return result;
}

/* ---------------------------------------------------------------------- */

void SurrogatePMI::write_query_batch(
    const std::string &path,
    const std::vector<std::vector<double>> &queries)
{
  if (queries.empty()) return;

  int n_queries = static_cast<int>(queries.size());
  int n_features = static_cast<int>(queries[0].size());

  // Flatten to row-major
  std::vector<double> flat(n_queries * n_features);
  for (int i = 0; i < n_queries; i++)
    for (int j = 0; j < n_features; j++)
      flat[i * n_features + j] = queries[i][j];

  try {
    H5::H5File file(path, H5F_ACC_TRUNC);
    hsize_t dims[2] = {static_cast<hsize_t>(n_queries),
                       static_cast<hsize_t>(n_features)};
    H5::DataSpace space(2, dims);
    H5::DataSet ds = file.createDataSet("queries",
                                         H5::PredType::NATIVE_DOUBLE,
                                         space);
    ds.write(flat.data(), H5::PredType::NATIVE_DOUBLE);
    file.close();
  } catch (const H5::Exception &) {
    // Silently fail on write errors (non-critical for simulation)
  }
}
