/* ----------------------------------------------------------------------
    OpenEdge: Lightweight NN surrogate for PMI inference
    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
      - Austin Nichols (ORNL, nicholsa@ornl.gov, 2025)
    https://github.com/ORNL-Fusion/OpenEdge

    Ensemble of small feed-forward neural networks predicting PMI
    coefficients (RN, RE, Y, implant_depth) from input features
    (E, theta, surface_composition).

    Architecture: 5 networks, each 3 hidden layers x 64 neurons, ReLU
    activation, linear output.  Trained by active_learning_server.py.

    Weights are loaded from an HDF5 file with layout:
      /ensemble/net0/layer0/weights  [n_in x 64]
      /ensemble/net0/layer0/biases   [64]
      /ensemble/net0/layer1/weights  [64 x 64]
      ...
      /ensemble/net0/output/weights  [64 x n_out]
      /ensemble/net0/output/biases   [n_out]

    Uncertainty is estimated from ensemble disagreement (standard
    deviation of predictions across networks).
------------------------------------------------------------------------- */

#ifndef SPARTA_SURROGATE_PMI_H
#define SPARTA_SURROGATE_PMI_H

#include <string>
#include <vector>

namespace SPARTA_NS {

// Output from the surrogate model
struct SurrogateOutput {
  double RN;              // reflection number coefficient
  double RE;              // reflected energy fraction
  double Y;               // sputtering yield
  double implant_depth;   // implantation depth (Angstroms)
  double uncertainty;     // ensemble disagreement (std dev of Y)
};

class SurrogatePMI {
 public:
  SurrogatePMI();
  ~SurrogatePMI();

  // Load weights from HDF5 file
  // Returns 0 on success, -1 on failure
  int load_weights(const std::string &path);

  // Evaluate the surrogate model
  // inputs: E (eV), theta (degrees), composition fractions
  SurrogateOutput evaluate(double E_eV, double theta_deg,
                            const std::vector<double> &composition) const;

  // Check if weights are loaded
  bool is_loaded() const { return loaded; }

  // Get number of input features
  int num_inputs() const { return n_input; }

  // Get number of networks in ensemble
  int num_networks() const { return n_ensemble; }

  // Write query batch to HDF5 for active learning
  static void write_query_batch(const std::string &path,
                                 const std::vector<std::vector<double>> &queries);

 private:
  bool loaded;
  int n_input;
  int n_output;
  int n_ensemble;
  int n_hidden;       // neurons per hidden layer
  int n_layers;       // number of hidden layers

  // Network weights: [ensemble][layer][row-major matrix or bias vector]
  struct Layer {
    std::vector<double> weights;  // [n_in * n_out]
    std::vector<double> biases;   // [n_out]
    int n_in, n_out;
  };

  struct Network {
    std::vector<Layer> hidden;
    Layer output;
  };

  std::vector<Network> ensemble;

  // Forward pass through a single network
  std::vector<double> forward(const Network &net,
                               const std::vector<double> &input) const;

  // ReLU activation
  static double relu(double x) { return x > 0.0 ? x : 0.0; }
};

}

#endif
