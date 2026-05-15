#pragma once

#include "ann_data.h"
#include <vector>
#include <string>

// LassoView: Label propagation for expanding user cell selection
// Adapted from lassoView.cpp (original pybind11 version)
class LassoView {
public:
    // Run label propagation on connectivity data
    // Returns expanded cell indices (including both user-selected and propagated)
    static std::vector<int> run(
        const AnnDataContainer::SparseCSR& connectivity,
        const std::vector<int>& obs_codes,     // category codes from obs column
        const std::vector<int>& selected_ids,   // user-selected cell indices
        int n_categories,                       // number of unique categories
        bool do_correct = true,
        double alpha = 0.5,
        int max_iter = 1000
    );
};
