#pragma once

#include <string>
#include <vector>
#include <map>
#include <variant>
#include <cstdint>
#include <algorithm>
#include <cctype>

// In-memory data model replacing Python AnnData
struct AnnDataContainer {
    // Basic info
    std::string dataset_name;
    int n_obs = 0;
    int n_vars = 0;

    // obsm: embedding key -> 2D float array [n_obs][dim]
    std::map<std::string, std::vector<std::vector<float>>> obsm;

    // obs column order (preserving insertion order)
    std::vector<std::string> obs_column_order;

    // obs: column name -> variant array
    // We store each column in the most natural type
    struct ObsColumn {
        enum Type { INT, FLOAT, STRING } type;
        std::vector<int> int_values;
        std::vector<float> float_values;
        std::vector<std::string> string_values;

        // For categorical columns, store codes + categories
        bool is_categorical = false;
        std::vector<int> cat_codes;         // index into cat_categories
        std::vector<std::string> cat_categories;
    };
    std::map<std::string, ObsColumn> obs;

    // obsp: sparse matrices (e.g., connectivities for LassoView)
    struct SparseCSR {
        std::vector<double> data;
        std::vector<int> indices;
        std::vector<int> indptr;
        int rows = 0;
        int cols = 0;
    };
    std::map<std::string, SparseCSR> obsp;

    // Source h5ad path (for Lasso-ARE subprocess)
    std::string source_h5ad_path;

    // Helpers
    bool empty() const { return n_obs == 0; }

    std::vector<std::string> getObsColumns() const {
        return obs_column_order;
    }

    std::vector<std::string> getAvailableEmbeddings() const {
        std::vector<std::string> result;
        for (const auto& [key, mat] : obsm) {
            if (!mat.empty() && mat[0].size() >= 2) {
                result.push_back(key);
            }
        }
        return result;
    }

    std::string defaultEmbeddingKey() const {
        static const std::vector<std::string> preferred = {"X_umap", "umap", "UMAP", "X_UMAP"};
        auto available = getAvailableEmbeddings();
        for (const auto& p : preferred) {
            for (const auto& a : available) {
                if (a == p) return a;
            }
        }
        // Case-insensitive fallback
        for (const auto& p : preferred) {
            for (const auto& a : available) {
                // Simple lowercase compare
                std::string a_lower = a;
                std::string p_lower = p;
                std::transform(a_lower.begin(), a_lower.end(), a_lower.begin(), ::tolower);
                std::transform(p_lower.begin(), p_lower.end(), p_lower.begin(), ::tolower);
                if (a_lower == p_lower) return a;
            }
        }
        // Fallback: any embedding containing "umap"
        for (const auto& a : available) {
            std::string a_lower = a;
            std::transform(a_lower.begin(), a_lower.end(), a_lower.begin(), ::tolower);
            if (a_lower.find("umap") != std::string::npos) return a;
        }
        return available.empty() ? "" : available[0];
    }

    std::string defaultColorColumn() const {
        static const std::vector<std::string> preferred = {"leiden", "level2", "level1"};
        for (const auto& p : preferred) {
            if (obs.count(p)) return p;
        }
        return obs_column_order.empty() ? "" : obs_column_order[0];
    }
};
