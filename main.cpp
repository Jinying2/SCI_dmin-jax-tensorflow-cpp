#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <fstream>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

int main() {
    // Load input coordinates
    std::vector<double> x_data(12);
    {
        std::ifstream fin("../x.txt");
        if (!fin) {
            std::cerr << "Failed to open x.txt, generating random input\n";
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(-2.0, 2.0);
            for (auto& val : x_data) val = dis(gen);
        } else {
            for (int i = 0; i < 12; ++i) fin >> x_data[i];
        }
    }
    
    std::cout << "Input x (12D):\n";
    for (double v : x_data) std::cout << v << " ";
    std::cout << "\n\n";

    // Load SavedModel
    const std::string export_dir = "../exported_model";
    tensorflow::SavedModelBundleLite bundle;
    tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;

    auto status = tensorflow::LoadSavedModel(session_options, run_options,
                                             export_dir,
                                             {tensorflow::kSavedModelTagServe},
                                             &bundle);
    if (!status.ok()) {
        std::cerr << "Failed to load model: " << status.ToString() << std::endl;
        return 1;
    }

    // Prepare input tensor
    tensorflow::Tensor input_tensor(tensorflow::DT_DOUBLE, tensorflow::TensorShape({12}));
    auto flat = input_tensor.flat<double>();
    for (int i = 0; i < 12; ++i) {
        flat(i) = x_data[i];
    }

    // Run model
    std::vector<tensorflow::Tensor> outputs;
    status = bundle.GetSession()->Run(
        {{"serving_default_args_tf_0:0", input_tensor}},
        {"PartitionedCall:0", "PartitionedCall:1"},
        {},
        &outputs
    );

    if (!status.ok()) {
        std::cerr << "Model execution failed: " << status.ToString() << std::endl;
        return 1;
    }

    // Print results
    std::cout << "Gradient (12D):\n";
    auto grad = outputs[0].flat<float>();
    for (int i = 0; i < grad.size(); ++i) {
        std::cout << std::fixed << std::setw(10) << std::setprecision(6) << grad(i) << " ";
        if ((i + 1) % 6 == 0) std::cout << "\n";
    }
    std::cout << "\n\n";

    std::cout << "Hessian (12x12):\n";
    auto hess = outputs[1].matrix<float>();
    for (int i = 0; i < hess.dimension(0); ++i) {
        for (int j = 0; j < hess.dimension(1); ++j) {
            std::cout << std::fixed << std::setw(10) << std::setprecision(6) << hess(i, j) << " ";
        }
        std::cout << "\n";
    }

    // Save results
    {
        std::ofstream fg("../cpp_grad.txt");
        fg << std::fixed << std::setprecision(6);
        for (int i = 0; i < 12; ++i) {
            fg << std::setw(12) << grad(i) << "\n";
        }
    }
    {
        std::ofstream fh("../cpp_hess.txt");
        fh << std::fixed << std::setprecision(6);
        for (int i = 0; i < 12; ++i) {
            for (int j = 0; j < 12; ++j) {
                fh << std::setw(12) << hess(i, j) << (j + 1 < 12 ? ' ' : '\n');
            }
        }
    }

    return 0;
}