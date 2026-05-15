#include "h5ad_reader.h"
#include <highfive/H5File.hpp>
#include <highfive/H5Group.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5Attribute.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cstring>

namespace hf = HighFive;

std::shared_ptr<AnnDataContainer> H5adReader::read(const std::string& path) {
    auto adata = std::make_shared<AnnDataContainer>();
    adata->source_h5ad_path = path;

    try {
        hf::File file(path, hf::File::ReadOnly);

        // Read n_obs and n_vars from shape attribute
        if (file.exist("shape")) {
            auto shape = file.getDataSet("shape").read<std::vector<int64_t>>();
            if (shape.size() >= 2) {
                adata->n_obs = static_cast<int>(shape[0]);
                adata->n_vars = static_cast<int>(shape[1]);
            }
        } else {
            // Infer from obs or X
            if (file.exist("obs") && file.getGroup("obs").exist("_index")) {
                auto idx = file.getGroup("obs").getDataSet("_index");
                adata->n_obs = static_cast<int>(idx.getDimensions()[0]);
            }
            if (file.exist("var") && file.getGroup("var").exist("_index")) {
                auto idx = file.getGroup("var").getDataSet("_index");
                adata->n_vars = static_cast<int>(idx.getDimensions()[0]);
            }
        }

        // Read obsm (embeddings)
        if (file.exist("obsm")) {
            readObsm(file, *adata);
        }

        // Read obs (cell metadata)
        if (file.exist("obs")) {
            readObs(file, *adata);
        }

        // Read obsp (sparse matrices like connectivities)
        if (file.exist("obsp")) {
            readObsp(file, *adata);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error reading h5ad file: " << e.what() << std::endl;
        return nullptr;
    }

    return adata;
}

void H5adReader::readObsm(hf::File& file, AnnDataContainer& adata) {
    auto obsm_group = file.getGroup("obsm");

    // Get keys from the group
    auto keys = obsm_group.listObjectNames();

    for (const auto& key : keys) {
        try {
            if (!obsm_group.exist(key)) continue;

            // Skip non-dataset entries (e.g., AUCell_rankings is a Group)
            auto obj_type = obsm_group.getObjectType(key);
            if (obj_type != hf::ObjectType::Dataset) continue;

            auto ds = obsm_group.getDataSet(key);
            auto dims = ds.getDimensions();
            if (dims.size() != 2 || dims[1] < 2) continue;

            int n = static_cast<int>(dims[0]);
            int d = static_cast<int>(dims[1]);

            // Read as 2D vector directly (HighFive requires matching dimensions)
            std::vector<std::vector<float>> coords;
            ds.read(coords);

            // If reading as 2D failed or returned wrong size, try reading as double
            if (static_cast<int>(coords.size()) != n || (n > 0 && static_cast<int>(coords[0].size()) != d)) {
                std::vector<std::vector<double>> coords_d;
                ds.read(coords_d);
                coords.resize(n, std::vector<float>(d));
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < d && j < static_cast<int>(coords_d[i].size()); j++) {
                        coords[i][j] = static_cast<float>(coords_d[i][j]);
                    }
                }
            }
            adata.obsm[key] = std::move(coords);

        } catch (const std::exception& e) {
            std::cerr << "Warning: could not read obsm/" << key << ": " << e.what() << std::endl;
        }
    }
}

void H5adReader::readObs(hf::File& file, AnnDataContainer& adata) {
    auto obs_group = file.getGroup("obs");
    auto col_names = obs_group.listObjectNames();

    for (const auto& col_name : col_names) {
        try {
            if (!obs_group.exist(col_name)) continue;

            // Check if it's a group (categorical) or dataset
            auto obj_type = obs_group.getObjectType(col_name);

            AnnDataContainer::ObsColumn column;

            if (obj_type == hf::ObjectType::Group) {
                // Categorical column: has "codes" and "categories" sub-datasets
                auto cat_group = obs_group.getGroup(col_name);

                if (cat_group.exist("codes") && cat_group.exist("categories")) {
                    column.is_categorical = true;

                    // Read codes
                    auto codes_ds = cat_group.getDataSet("codes");
                    codes_ds.read(column.cat_codes);

                    // Read categories
                    auto cats_ds = cat_group.getDataSet("categories");
                    // Categories can be string or numeric
                    auto cat_type = cats_ds.getDataType();
                    if (cat_type.getClass() == hf::DataTypeClass::String) {
                        cats_ds.read(column.cat_categories);
                    } else {
                        // Numeric categories - convert to string
                        std::vector<int> int_cats;
                        cats_ds.read(int_cats);
                        for (auto v : int_cats) {
                            column.cat_categories.push_back(std::to_string(v));
                        }
                    }

                    // Build string values from codes + categories
                    column.type = AnnDataContainer::ObsColumn::STRING;
                    column.string_values.resize(column.cat_codes.size());
                    for (size_t i = 0; i < column.cat_codes.size(); i++) {
                        int code = column.cat_codes[i];
                        if (code >= 0 && code < (int)column.cat_categories.size()) {
                            column.string_values[i] = column.cat_categories[code];
                        } else {
                            column.string_values[i] = "";
                        }
                    }
                }
            } else if (obj_type == hf::ObjectType::Dataset) {
                auto ds = obs_group.getDataSet(col_name);
                auto dtype = ds.getDataType();

                if (col_name == "_index") {
                    // This is the cell barcode index, skip it as a regular column
                    // but store cell_ids if needed
                    continue;
                }

                if (dtype.getClass() == hf::DataTypeClass::String) {
                    column.type = AnnDataContainer::ObsColumn::STRING;
                    ds.read(column.string_values);
                } else if (dtype.getClass() == hf::DataTypeClass::Integer) {
                    column.type = AnnDataContainer::ObsColumn::INT;
                    // Try reading as int first, fall back to int64
                    try {
                        ds.read(column.int_values);
                    } catch (...) {
                        std::vector<int64_t> tmp;
                        ds.read(tmp);
                        column.int_values.reserve(tmp.size());
                        for (auto v : tmp) column.int_values.push_back(static_cast<int>(v));
                    }
                } else if (dtype.getClass() == hf::DataTypeClass::Float) {
                    column.type = AnnDataContainer::ObsColumn::FLOAT;
                    // Try reading as float first, fall back to double
                    try {
                        ds.read(column.float_values);
                    } catch (...) {
                        std::vector<double> tmp;
                        ds.read(tmp);
                        column.float_values.reserve(tmp.size());
                        for (auto v : tmp) column.float_values.push_back(static_cast<float>(v));
                    }
                } else {
                    continue; // Skip unsupported types
                }
            }

            adata.obs[col_name] = std::move(column);
            adata.obs_column_order.push_back(col_name);

        } catch (const std::exception& e) {
            std::cerr << "Warning: could not read obs/" << col_name << ": " << e.what() << std::endl;
        }
    }
}

void H5adReader::readObsp(hf::File& file, AnnDataContainer& adata) {
    auto obsp_group = file.getGroup("obsp");
    auto keys = obsp_group.listObjectNames();

    for (const auto& key : keys) {
        try {
            if (!obsp_group.exist(key)) continue;
            auto obj_type = obsp_group.getObjectType(key);
            if (obj_type != hf::ObjectType::Group) continue;

            auto mat_group = obsp_group.getGroup(key);

            // CSR sparse matrix format: data, indices, indptr + shape attribute
            if (!mat_group.exist("data") || !mat_group.exist("indices") || !mat_group.exist("indptr")) {
                continue;
            }

            AnnDataContainer::SparseCSR csr;

            // Read data
            auto data_ds = mat_group.getDataSet("data");
            data_ds.read(csr.data);

            // Read indices (int32 or int64)
            auto idx_ds = mat_group.getDataSet("indices");
            auto idx_type = idx_ds.getDataType();
            if (idx_type.getSize() == 8) {
                std::vector<int64_t> tmp;
                idx_ds.read(tmp);
                csr.indices.reserve(tmp.size());
                for (auto v : tmp) csr.indices.push_back(static_cast<int>(v));
            } else {
                std::vector<int32_t> tmp;
                idx_ds.read(tmp);
                csr.indices.reserve(tmp.size());
                for (auto v : tmp) csr.indices.push_back(static_cast<int>(v));
            }

            // Read indptr
            auto indptr_ds = mat_group.getDataSet("indptr");
            auto indptr_type = indptr_ds.getDataType();
            if (indptr_type.getSize() == 8) {
                std::vector<int64_t> tmp;
                indptr_ds.read(tmp);
                csr.indptr.reserve(tmp.size());
                for (auto v : tmp) csr.indptr.push_back(static_cast<int>(v));
            } else {
                std::vector<int32_t> tmp;
                indptr_ds.read(tmp);
                csr.indptr.reserve(tmp.size());
                for (auto v : tmp) csr.indptr.push_back(static_cast<int>(v));
            }

            // Read shape attribute
            if (mat_group.hasAttribute("shape")) {
                auto shape_attr = mat_group.getAttribute("shape");
                std::vector<int64_t> shape;
                shape_attr.read(shape);
                if (shape.size() >= 2) {
                    csr.rows = static_cast<int>(shape[0]);
                    csr.cols = static_cast<int>(shape[1]);
                }
            } else {
                csr.rows = static_cast<int>(csr.indptr.size()) - 1;
                csr.cols = csr.rows; // Assume square
            }

            adata.obsp[key] = std::move(csr);

        } catch (const std::exception& e) {
            std::cerr << "Warning: could not read obsp/" << key << ": " << e.what() << std::endl;
        }
    }
}

// --- Matrix file reading (alternative to h5ad) ---

std::shared_ptr<AnnDataContainer> H5adReader::readFromMatrixFiles(const std::string& folder_path) {
    auto adata = std::make_shared<AnnDataContainer>();
    adata->source_h5ad_path = folder_path;

    // Read UMAP coordinates from umap.txt (tab-separated, n_obs x 2)
    std::string umap_path = folder_path + "/umap.txt";
    std::ifstream umap_file(umap_path);
    if (umap_file.is_open()) {
        std::vector<std::vector<float>> umap_coords;
        std::string line;
        while (std::getline(umap_file, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            float x, y;
            if (iss >> x >> y) {
                umap_coords.push_back({x, y});
            }
        }
        adata->obsm["X_umap"] = std::move(umap_coords);
        adata->n_obs = static_cast<int>(adata->obsm["X_umap"].size());
    }

    // Read obs.csv (header + data rows)
    std::string obs_path = folder_path + "/obs.csv";
    std::ifstream obs_file(obs_path);
    if (obs_file.is_open()) {
        std::string header_line;
        if (std::getline(obs_file, header_line)) {
            // Parse header
            std::vector<std::string> headers;
            std::istringstream hss(header_line);
            std::string col;
            while (std::getline(hss, col, ',')) {
                headers.push_back(col);
            }

            // Initialize columns
            std::vector<AnnDataContainer::ObsColumn> columns(headers.size());
            for (auto& c : columns) {
                c.type = AnnDataContainer::ObsColumn::STRING;
            }

            // Read data rows
            std::string line;
            while (std::getline(obs_file, line)) {
                if (line.empty()) continue;
                std::istringstream iss(line);
                std::string val;
                int col_idx = 0;
                while (std::getline(iss, val, ',') && col_idx < (int)headers.size()) {
                    columns[col_idx].string_values.push_back(val);
                    col_idx++;
                }
            }

            // Store columns
            for (size_t i = 0; i < headers.size(); i++) {
                adata->obs[headers[i]] = std::move(columns[i]);
                adata->obs_column_order.push_back(headers[i]);
            }
        }
    }

    // Read connectivities.mtx (MatrixMarket format, optional)
    std::string mtx_path = folder_path + "/connectivities.mtx";
    std::ifstream mtx_file(mtx_path);
    if (mtx_file.is_open()) {
        // Skip comment lines
        std::string line;
        while (std::getline(mtx_file, line)) {
            if (line.empty() || line[0] != '%') break;
        }

        // Parse dimensions line
        int n_rows, n_cols, n_nonzero;
        sscanf(line.c_str(), "%d %d %d", &n_rows, &n_cols, &n_nonzero);

        // Read COO entries and convert to CSR
        AnnDataContainer::SparseCSR csr;
        csr.rows = n_rows;
        csr.cols = n_cols;
        csr.indptr.resize(n_rows + 1, 0);

        struct COOEntry { int row, col; double val; };
        std::vector<COOEntry> entries;
        entries.reserve(n_nonzero);

        while (std::getline(mtx_file, line)) {
            if (line.empty()) continue;
            int r, c;
            double v;
            sscanf(line.c_str(), "%d %d %lf", &r, &c, &v);
            entries.push_back({r - 1, c - 1, v}); // MTX is 1-indexed
            csr.indptr[r]++; // Count entries per row
        }

        // Convert to CSR
        for (int i = 0; i < n_rows; i++) {
            csr.indptr[i + 1] += csr.indptr[i];
        }
        // Shift indptr
        for (int i = n_rows; i > 0; i--) {
            csr.indptr[i] = csr.indptr[i - 1];
        }
        csr.indptr[0] = 0;

        csr.data.resize(entries.size());
        csr.indices.resize(entries.size());
        std::vector<int> pos(n_rows, 0);

        for (const auto& e : entries) {
            int idx = csr.indptr[e.row] + pos[e.row];
            csr.data[idx] = e.val;
            csr.indices[idx] = e.col;
            pos[e.row]++;
        }

        adata->obsp["connectivities"] = std::move(csr);
    }

    if (adata->n_obs == 0 && !adata->obs.empty()) {
        adata->n_obs = static_cast<int>(adata->obs.begin()->second.string_values.size());
    }

    return adata;
}
