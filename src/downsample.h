#pragma once

#include "ann_data.h"
#include <vector>
#include <string>
#include <memory>

// Downsample: C++ port of do_downsample.py
class Downsample {
public:
    struct Result {
        std::shared_ptr<AnnDataContainer> sampled_adata;
        std::vector<int> nearest_ids;  // For each original cell, index in downsampled set
    };

    // Run downsampling
    static Result run(
        std::shared_ptr<AnnDataContainer> adata,
        double sample_rate = 0.1,
        double leiden_r = 1.0,
        double uniform_rate = 0.5,
        const std::string& cluster_key = "leiden",
        const std::string& obsm_key = "X_umap"
    );

    // Recover full selection from downsampled selection
    static std::vector<int> recoverSelection(
        const std::vector<int>& selected_in_downsampled,
        const std::vector<int>& nearest_ids
    );
};
