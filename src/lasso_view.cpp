#include "lasso_view.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_map>

// ============================================================
// Core label propagation algorithm (adapted from lassoView.cpp)
// Original: pybind11-bound C++ implementation
// Adapted: standalone C++ with Qt integration
// ============================================================

// --- Sparse COO matrix ---
class MatCoo {
public:
    struct Elem {
        double v;
        int row, col;
        bool operator<(const Elem& other) const {
            return row < other.row || (row == other.row && col < other.col);
        }
        bool operator<(int i) const {
            return row < i;
        }
    };
    std::vector<Elem> elem;
    int n, m;
    int totalElements = 0;

    MatCoo() : n(0), m(0) {}
    MatCoo(int n0, int m0) : n(n0), m(m0) {}

    void append(int r, int c, double val) {
        elem.push_back({val, r, c});
        totalElements++;
    }
};

// --- Dense matrix ---
class Mat {
public:
    std::vector<double> data;
    int n, m;

    Mat() : n(0), m(0) {}
    Mat(int n0, int m0) : n(n0), m(m0), data(n0 * m0, 0) {}

    double getVal(int i, int j) const {
        return data[i * m + j];
    }

    void setVal(int i, int j, double v) {
        data[i * m + j] = v;
    }

    void setNeg() {
        std::fill(data.begin(), data.end(), -1.0);
    }

    void editVal2(int i, int j, double v) {
        data[i * m + j] = v;
    }
};

// --- Label propagation ---
static void labelPropagation(MatCoo& X, Mat& y_label, Mat& y_pred, Mat& y_new,
                              double alpha, int max_iter) {
    int n = X.n;
    int K = y_label.m;

    // Build row-normalized transition matrix from COO
    // For each row, compute sum of weights and normalize
    std::vector<double> row_sum(n, 0);
    for (const auto& e : X.elem) {
        if (e.row >= 0 && e.row < n) {
            row_sum[e.row] += e.v;
        }
    }

    // Iterative propagation
    // y_pred = y_label (initial)
    y_pred.data = y_label.data;
    y_new.data = y_label.data;

    for (int iter = 0; iter < max_iter; iter++) {
        // y_new = alpha * T * y_pred + (1 - alpha) * y_label
        std::fill(y_new.data.begin(), y_new.data.end(), 0);

        for (const auto& e : X.elem) {
            if (e.row >= 0 && e.row < n && row_sum[e.row] > 0) {
                double norm_val = e.v / row_sum[e.row];
                for (int k = 0; k < K; k++) {
                    y_new.setVal(e.row, k, y_new.getVal(e.row, k) + norm_val * y_pred.getVal(e.col, k));
                }
            }
        }

        for (int i = 0; i < n; i++) {
            for (int k = 0; k < K; k++) {
                double v = alpha * y_new.getVal(i, k) + (1 - alpha) * y_label.getVal(i, k);
                y_new.setVal(i, k, v);
            }
        }

        // Check convergence
        double diff = 0;
        for (int i = 0; i < n * K; i++) {
            diff += std::abs(y_new.data[i] - y_pred.data[i]);
        }
        if (diff < 1e-6) break;

        y_pred.data = y_new.data;
    }
}

// --- Rectification step ---
static void rectify(Mat& y_label, Mat& y_pred, Mat& y_new, int n, int K) {
    for (int i = 0; i < n; i++) {
        int max_k = 0;
        double max_v = y_new.getVal(i, 0);
        for (int k = 1; k < K; k++) {
            if (y_new.getVal(i, k) > max_v) {
                max_v = y_new.getVal(i, k);
                max_k = k;
            }
        }
        // Set the predicted class to the max
        for (int k = 0; k < K; k++) {
            y_new.setVal(i, k, (k == max_k) ? 1.0 : 0.0);
        }
    }

    // Re-propagate with rectified values
    // (In the original code this is done by iterating again)
    // For simplicity, we just use the rectified output
}

// --- Public API ---
std::vector<int> LassoView::run(
    const AnnDataContainer::SparseCSR& connectivity,
    const std::vector<int>& obs_codes,
    const std::vector<int>& selected_ids,
    int n_categories,
    bool do_correct,
    double alpha,
    int max_iter)
{
    int n = connectivity.rows;
    if (n == 0) n = static_cast<int>(obs_codes.size());
    int K = n_categories + 1; // +1 for the new selected category

    // 1. Convert CSR connectivity to COO format for MatCoo
    MatCoo X(n, n);
    for (int row = 0; row < n; row++) {
        int start = connectivity.indptr[row];
        int end = (row + 1 < (int)connectivity.indptr.size()) ? connectivity.indptr[row + 1] : (int)connectivity.data.size();
        for (int idx = start; idx < end; idx++) {
            X.append(row, connectivity.indices[idx], connectivity.data[idx]);
        }
    }

    // 2. Build category mapping
    std::unordered_map<int, int> val_map;
    for (int code : obs_codes) {
        if (val_map.find(code) == val_map.end()) {
            val_map[code] = static_cast<int>(val_map.size());
        }
    }
    int selected_val = K - 1; // New category for selected cells

    // 3. Build y_label matrix
    Mat y_label(n, K);
    y_label.setNeg(); // Initialize all to -1

    // Random 10% labeled
    std::vector<int> select_list(n, 0);
    std::mt19937 rng(42);
    int n_random = std::max(1, n / 10);
    std::vector<int> all_indices(n);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::shuffle(all_indices.begin(), all_indices.end(), rng);
    for (int i = 0; i < n_random && i < (int)all_indices.size(); i++) {
        select_list[all_indices[i]] = 1;
    }

    // Set selected cells
    std::vector<int> modified_codes = obs_codes;
    for (int idx : selected_ids) {
        if (idx >= 0 && idx < n) {
            select_list[idx] = 1;
            modified_codes[idx] = selected_val;
        }
    }

    // Build label matrix
    for (int i = 0; i < n; i++) {
        if (select_list[i]) {
            auto it = val_map.find(modified_codes[i]);
            int mapped_val = (modified_codes[i] == selected_val) ? selected_val :
                             (it != val_map.end() ? it->second : 0);
            y_label.editVal2(i, mapped_val, 1.0);
        }
    }

    // 4. Run label propagation
    Mat y_pred(n, K);
    Mat y_new(n, K);
    labelPropagation(X, y_label, y_pred, y_new, alpha, max_iter);

    // 5. Apply rectification if needed
    if (do_correct) {
        rectify(y_label, y_pred, y_new, n, K);
    }

    // 6. Collect results: all cells where the predicted category == selected_val
    std::vector<int> result;
    for (int i = 0; i < n; i++) {
        double v = do_correct ? y_new.getVal(i, selected_val) : y_pred.getVal(i, selected_val);
        if (v > 0) {
            result.push_back(i);
        }
    }

    return result;
}
