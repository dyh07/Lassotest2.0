#include "downsample.h"
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_map>
#include <map>
#include <set>
#include <limits>

Downsample::Result Downsample::run(
    std::shared_ptr<AnnDataContainer> adata,
    double sample_rate,
    double leiden_r,
    double uniform_rate,
    const std::string& cluster_key,
    const std::string& obsm_key)
{
    Result result;

    // Get UMAP coordinates
    std::string emb_key = obsm_key;
    if (adata->obsm.find(emb_key) == adata->obsm.end()) {
        // Try to find any UMAP embedding
        for (auto& [k, v] : adata->obsm) {
            std::string k_lower = k;
            std::transform(k_lower.begin(), k_lower.end(), k_lower.begin(), ::tolower);
            if (k_lower.find("umap") != std::string::npos) {
                emb_key = k;
                break;
            }
        }
    }

    auto& umap_coords = adata->obsm[emb_key];
    int n_obs = adata->n_obs;

    // Get cluster labels
    std::vector<int> cluster_labels(n_obs, 0);
    int n_clusters = 1;
    if (adata->obs.find(cluster_key) != adata->obs.end()) {
        auto& col = adata->obs[cluster_key];
        if (col.is_categorical && !col.cat_codes.empty()) {
            cluster_labels = col.cat_codes;
            n_clusters = *std::max_element(cluster_labels.begin(), cluster_labels.end()) + 1;
        } else if (col.type == AnnDataContainer::ObsColumn::STRING && !col.string_values.empty()) {
            // Map string categories to integers
            std::map<std::string, int> cat_map;
            for (int i = 0; i < n_obs; i++) {
                auto it = cat_map.find(col.string_values[i]);
                if (it == cat_map.end()) {
                    cat_map[col.string_values[i]] = static_cast<int>(cat_map.size());
                }
                cluster_labels[i] = cat_map[col.string_values[i]];
            }
            n_clusters = static_cast<int>(cat_map.size());
        } else if (col.type == AnnDataContainer::ObsColumn::INT && !col.int_values.empty()) {
            cluster_labels = col.int_values;
            n_clusters = *std::max_element(cluster_labels.begin(), cluster_labels.end()) + 1;
        }
    }

    // Calculate sample sizes
    uniform_rate = std::max(0.0, std::min(1.0, uniform_rate));
    int sample_cells_num = std::min(
        static_cast<int>(n_obs * sample_rate) + n_clusters * 3,
        n_obs
    );
    int n_uniform = static_cast<int>(sample_cells_num * uniform_rate);
    int n_balanced = sample_cells_num - n_uniform;

    std::mt19937 rng(42);

    // 1. Uniform sampling
    std::vector<int> uniform_indices;
    if (n_uniform > 0) {
        std::vector<int> all_idx(n_obs);
        std::iota(all_idx.begin(), all_idx.end(), 0);
        std::shuffle(all_idx.begin(), all_idx.end(), rng);
        uniform_indices.assign(all_idx.begin(), all_idx.begin() + std::min(n_uniform, n_obs));
    }

    // 2. Balanced per-cluster sampling
    std::vector<int> balanced_indices;
    if (n_balanced > 0) {
        // Group indices by cluster
        std::map<int, std::vector<int>> cluster_to_indices;
        for (int i = 0; i < n_obs; i++) {
            cluster_to_indices[cluster_labels[i]].push_back(i);
        }

        int n_per_cluster = std::max(1, n_balanced / n_clusters);

        for (auto& [cluster, indices] : cluster_to_indices) {
            int n_sample = std::min(static_cast<int>(indices.size()), n_per_cluster);
            if (n_sample > 0) {
                std::shuffle(indices.begin(), indices.end(), rng);
                for (int i = 0; i < n_sample; i++) {
                    balanced_indices.push_back(indices[i]);
                }
            }
        }
    }

    // 3. Combine and deduplicate
    std::vector<int> final_indices;
    final_indices.reserve(uniform_indices.size() + balanced_indices.size());
    final_indices.insert(final_indices.end(), uniform_indices.begin(), uniform_indices.end());
    final_indices.insert(final_indices.end(), balanced_indices.begin(), balanced_indices.end());
    std::sort(final_indices.begin(), final_indices.end());
    final_indices.erase(std::unique(final_indices.begin(), final_indices.end()), final_indices.end());

    // 4. Create sampled adata
    result.sampled_adata = std::make_shared<AnnDataContainer>();
    result.sampled_adata->dataset_name = adata->dataset_name + " (downsampled)";
    result.sampled_adata->n_obs = static_cast<int>(final_indices.size());
    result.sampled_adata->n_vars = adata->n_vars;
    result.sampled_adata->source_h5ad_path = adata->source_h5ad_path;

    // Copy obsm for sampled cells
    for (auto& [key, coords] : adata->obsm) {
        auto& sampled_coords = result.sampled_adata->obsm[key];
        int dim = coords.empty() ? 0 : static_cast<int>(coords[0].size());
        sampled_coords.resize(final_indices.size());
        for (size_t i = 0; i < final_indices.size(); i++) {
            sampled_coords[i] = coords[final_indices[i]];
        }
    }

    // Copy obs for sampled cells
    for (auto& [key, col] : adata->obs) {
        auto& sampled_col = result.sampled_adata->obs[key];
        sampled_col.type = col.type;
        sampled_col.is_categorical = col.is_categorical;
        sampled_col.cat_categories = col.cat_categories;

        if (col.type == AnnDataContainer::ObsColumn::STRING && !col.string_values.empty()) {
            sampled_col.string_values.resize(final_indices.size());
            for (size_t i = 0; i < final_indices.size(); i++) {
                sampled_col.string_values[i] = col.string_values[final_indices[i]];
            }
        } else if (col.type == AnnDataContainer::ObsColumn::INT && !col.int_values.empty()) {
            sampled_col.int_values.resize(final_indices.size());
            for (size_t i = 0; i < final_indices.size(); i++) {
                sampled_col.int_values[i] = col.int_values[final_indices[i]];
            }
        } else if (col.type == AnnDataContainer::ObsColumn::FLOAT && !col.float_values.empty()) {
            sampled_col.float_values.resize(final_indices.size());
            for (size_t i = 0; i < final_indices.size(); i++) {
                sampled_col.float_values[i] = col.float_values[final_indices[i]];
            }
        }

        if (col.is_categorical && !col.cat_codes.empty()) {
            sampled_col.cat_codes.resize(final_indices.size());
            for (size_t i = 0; i < final_indices.size(); i++) {
                sampled_col.cat_codes[i] = col.cat_codes[final_indices[i]];
            }
        }
    }
    result.sampled_adata->obs_column_order = adata->obs_column_order;

    // Add orig_idx column
    {
        AnnDataContainer::ObsColumn orig_idx_col;
        orig_idx_col.type = AnnDataContainer::ObsColumn::INT;
        orig_idx_col.int_values = final_indices;
        // Convert to string for display
        orig_idx_col.string_values.resize(final_indices.size());
        for (size_t i = 0; i < final_indices.size(); i++) {
            orig_idx_col.string_values[i] = std::to_string(final_indices[i]);
        }
        result.sampled_adata->obs["orig_idx"] = std::move(orig_idx_col);
        result.sampled_adata->obs_column_order.push_back("orig_idx");
    }

    // 5. Find nearest neighbors (brute-force in UMAP space)
    auto& sampled_umap = result.sampled_adata->obsm[emb_key];
    result.nearest_ids.resize(n_obs);

    for (int i = 0; i < n_obs; i++) {
        float ox = umap_coords[i][0];
        float oy = umap_coords[i][1];
        float min_dist = std::numeric_limits<float>::max();
        int min_idx = 0;
        for (size_t j = 0; j < sampled_umap.size(); j++) {
            float dx = ox - sampled_umap[j][0];
            float dy = oy - sampled_umap[j][1];
            float dist = dx * dx + dy * dy;
            if (dist < min_dist) {
                min_dist = dist;
                min_idx = static_cast<int>(j);
            }
        }
        result.nearest_ids[i] = min_idx;
    }

    return result;
}

std::vector<int> Downsample::recoverSelection(
    const std::vector<int>& selected_in_downsampled,
    const std::vector<int>& nearest_ids)
{
    std::set<int> selected_set(selected_in_downsampled.begin(), selected_in_downsampled.end());
    std::vector<int> recovered;
    for (int i = 0; i < static_cast<int>(nearest_ids.size()); i++) {
        if (selected_set.count(nearest_ids[i])) {
            recovered.push_back(i);
        }
    }
    return recovered;
}
