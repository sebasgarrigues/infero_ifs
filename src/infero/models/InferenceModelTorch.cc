#include <iostream>

#include <algorithm>

#include "infero/models/InferenceModelTorch.h"
#include "infero/infero_utils.h"

namespace infero {


static InferenceModelBuilder<InferenceModelTorch> torchBuilder;

eckit::LocalConfiguration InferenceModelTorch::defaultConfig() {
    static eckit::LocalConfiguration config;
    // empty defaults..
    return config;
}


InferenceModelTorch::InferenceModelTorch(const eckit::Configuration& conf) : 
    InferenceModel(conf, InferenceModelTorch::defaultConfig()){

    // read/bcast model by mpi (when possible)
    broadcast_model(modelPath());

    // Load the model    
    try {
        eckit::Log::info() << "Loading model from: " << modelPath() << std::endl;
        torch_module_ = torch::jit::load(modelPath());
    } catch (const c10::Error& e) {
        eckit::Log::error() << "Error loading the model: " << modelPath() << " - " << e.what() << std::endl;
        throw eckit::SeriousBug("Error loading the model");
    }

}


InferenceModelTorch::~InferenceModelTorch() {}


std::string InferenceModelTorch::name() const {
    return "InferenceModelTorch";
}


void InferenceModelTorch::print(std::ostream& os) const {
    os << "InferenceModelTorch";
}

void InferenceModelTorch::infer_impl(eckit::linalg::TensorFloat& tIn, eckit::linalg::TensorFloat& tOut,
                                        std::string input_name, std::string output_name) {


    eckit::Log::info() << "--> Input tensor: " << tIn << std::endl;

    // Load the input tensor
    auto shape = tIn.shape();
    std::vector<int64_t> shape_int64(shape.size());
    std::copy(shape.begin(), shape.end(), shape_int64.begin());
    
    torch::IntArrayRef shape_ref(shape_int64.data(), shape_int64.size());
    torch::Tensor input = torch::from_blob(tIn.data(), shape_ref);

    // Pass input tensor to the model
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    // Run inference
    at::Tensor out_tensor;
    try {
        auto output = torch_module_.forward(inputs);
        out_tensor = output.toTensor();
    } catch (const c10::Error& e) {
        std::cerr << "Error running inference" << std::endl;
    }

    // --- Copy data from the output tensor ---
    at::IntArrayRef sizes = out_tensor.sizes();
    std::vector<int64_t> out_shape(sizes.begin(), sizes.end());
    if (tOut.layout() == eckit::linalg::TensorFloat::Layout::ColMajor) {

         // ONNX uses Left (C) tensor layouts, so we need to convert
         eckit::linalg::TensorFloat tLeft(out_tensor.data_ptr<float>(),
                                          utils::convert_shape<int64_t, size_t>(out_shape),
                                          eckit::linalg::TensorFloat::Layout::RowMajor);
         eckit::linalg::TensorFloat tRight = tLeft.transformRowMajorToColMajor();
         tOut = tRight;
    } else {
         // ONNX uses Left (C) tensor layouts, so we can copy straight into memory of tOut
         memcpy(tOut.data(), out_tensor.data_ptr<float>(), out_tensor.numel() * sizeof(float));
    }

    eckit::Log::info() << "--> Output tensor: " << tOut << std::endl;

}

void InferenceModelTorch::infer_mimo_impl(std::vector<eckit::linalg::TensorFloat*> &tIn, std::vector<const char*> &input_names,
                                            std::vector<eckit::linalg::TensorFloat*> &tOut, std::vector<const char*> &output_names) {}


} // namespace infero